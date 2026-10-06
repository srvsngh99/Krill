import Foundation
@preconcurrency import AVFoundation
import CoreMedia
import CoreVideo

// MARK: - EmbeddingGemma 2: video
//
// A video has no tower of its own. The reference
// (`EmbeddingGemma2VideoProcessor` + `EmbeddingGemma2Processor`) does:
//
//   1. sample frames: `fps = 1`, `max_frames = 32`, `overflow_strategy = uniform`
//        step        = native_fps / 1
//        num_sampled = max(1, int(duration * 1))
//        indices     = [min(total_frames - 1, int(i * step)) for i in range(num_sampled)]
//        if len(indices) > 32: indices = indices[linspace(0, len-1, 32, dtype=int)]
//   2. per frame: aspect-preserving resize to the 140-soft-token patch budget
//      (`max_soft_tokens: 140`, i.e. 1260 patches) with antialiased bicubic,
//      rescale 1/255, no normalisation  (= `EG2ImagePreprocessor.prepare(.., 140)`)
//   3. the SAME vision tower + `embed_vision` as images (`EG2VisionTower`)
//   4. layout: one `<boi> <|video|>xN <eoi>` block per frame, concatenated, NO
//      timestamps (`add_timestamps: false`, the checkpoint default) and no
//      separators; the audio track of a video is never read.
//
// This file adds the frame sampler (a port of step 1) and an AVFoundation
// decoder that returns exactly the sampled frames.

public enum EG2VideoError: Error, CustomStringConvertible, Equatable {
    case undecodable(String)
    case noVideoTrack
    case noFrames
    case platformUnavailable

    public var description: String {
        switch self {
        case .undecodable(let why):
            return "video could not be decoded (\(why)); AVFoundation reads mp4, mov and m4v (H.264 / HEVC)"
        case .noVideoTrack: return "the file has no video track"
        case .noFrames: return "the video has no decodable frames"
        case .platformUnavailable: return "video decoding is not available on this platform"
        }
    }
}

/// What the sampler needs to know about a video (torchcodec's `VideoMetadata`).
public struct EG2VideoInfo: Equatable, Sendable {
    public let totalFrames: Int
    /// Native (average) frame rate.
    public let fps: Double
    public let duration: Double
    /// Pixel size of a decoded frame.
    public let width: Int
    public let height: Int
    public init(totalFrames: Int, fps: Double, duration: Double, width: Int, height: Int) {
        self.totalFrames = totalFrames; self.fps = fps; self.duration = duration
        self.width = width; self.height = height
    }
}

public enum EG2VideoSampler {
    public static let targetFps = 1.0
    public static let maxFrames = 32

    /// Port of `EmbeddingGemma2VideoProcessor.sample_frames` (fps + `uniform`
    /// overflow). `fps` / `duration` are the file's native rate and length.
    public static func frameIndices(totalFrames: Int, fps: Double, duration: Double,
                                    targetFps: Double = targetFps,
                                    maxFrames: Int = maxFrames) -> [Int] {
        guard totalFrames > 0, fps > 0, duration > 0 else { return [] }
        let step = fps / targetFps  // native frames per sampled frame
        let numSampled = max(1, Int(duration * targetFps))
        var idx = (0 ..< numSampled).map { min(totalFrames - 1, Int(Double($0) * step)) }
        if idx.count > maxFrames {
            // np.linspace(0, n - 1, max_frames, dtype=int): i * step, last point pinned, truncated.
            let n = idx.count
            let s = Double(n - 1) / Double(maxFrames - 1)
            idx = (0 ..< maxFrames).map { i in
                let pos = i == maxFrames - 1 ? Double(n - 1) : Double(i) * s
                return idx[Int(pos)]
            }
        }
        return idx
    }
}

/// A video written to a temp file for AVFoundation, with its metadata.
/// The file is deleted when the object goes away.
public final class EG2VideoSource {
    public let info: EG2VideoInfo
    let url: URL
    private let dir: URL

    /// Open `data` (mp4 / mov / m4v). Reads the container only: counts compressed
    /// frames (no decode) and reads the track's rate, length and size.
    public init(data: Data, formatHint: String? = nil) throws {
        guard !data.isEmpty else { throw EG2VideoError.undecodable("empty") }
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-eg2-video-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        dir = d
        let ext = Self.fileExtension(hint: formatHint, data: data)
        url = d.appendingPathComponent("input.\(ext)")
        do {
            try data.write(to: url, options: .atomic)
            info = try Self.inspect(url)
        } catch {
            try? FileManager.default.removeItem(at: d)
            throw error
        }
    }

    deinit { try? FileManager.default.removeItem(at: dir) }

    static func fileExtension(hint: String?, data: Data) -> String {
        var h = (hint ?? "").lowercased()
        if let slash = h.lastIndex(of: "/") { h = String(h[h.index(after: slash)...]) }
        switch h {
        case "quicktime", "mov": return "mov"
        case "x-m4v", "m4v": return "m4v"
        case "mp4": return "mp4"
        default:
            let b = [UInt8](data.prefix(12))
            if b.count >= 12, Array(b[4 ..< 8]) == Array("ftyp".utf8) {
                return Array(b[8 ..< 11]) == Array("qt ".utf8) ? "mov" : "mp4"
            }
            return "mp4"
        }
    }

    private static func inspect(_ url: URL) throws -> EG2VideoInfo {
        let asset = AVURLAsset(url: url)
        guard asset.isReadable else { throw EG2VideoError.undecodable("not readable") }
        // Synchronous track accessors (deprecated in favour of async `load`, but this
        // runs on the embedding engine's synchronous path).
        guard let track = asset.tracks(withMediaType: .video).first else { throw EG2VideoError.noVideoTrack }
        let natural = track.naturalSize
        let nominal = track.nominalFrameRate
        let trackRange = track.timeRange

        // Count compressed samples (one per frame): no decoding.
        let reader: AVAssetReader
        do { reader = try AVAssetReader(asset: asset) } catch { throw EG2VideoError.undecodable(error.localizedDescription) }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        out.alwaysCopiesSampleData = false
        guard reader.canAdd(out) else { throw EG2VideoError.undecodable("cannot read the video track") }
        reader.add(out)
        guard reader.startReading() else {
            throw EG2VideoError.undecodable(reader.error?.localizedDescription ?? "cannot start reading")
        }
        var frames = 0
        while let sb = out.copyNextSampleBuffer() { frames += CMSampleBufferGetNumSamples(sb) }
        guard frames > 0 else { throw EG2VideoError.noFrames }

        let duration = CMTimeGetSeconds(trackRange.duration)
        guard duration.isFinite, duration > 0 else { throw EG2VideoError.undecodable("unknown duration") }
        let fps = nominal > 0 ? Double(nominal) : Double(frames) / duration
        let w = Int(natural.width.rounded()), h = Int(natural.height.rounded())
        guard w > 0, h > 0 else { throw EG2VideoError.undecodable("unknown frame size") }
        return EG2VideoInfo(totalFrames: frames, fps: fps, duration: duration, width: w, height: h)
    }

    /// Decode the frames at `indices` (ascending, display order) and map each
    /// through `transform` as it is produced, so only the transformed result is
    /// kept (a 4K frame is never held 32 times). Audio is never touched.
    public func frames<T>(at indices: [Int], _ transform: (EG2RGBImage) throws -> T) throws -> [T] {
        guard let last = indices.last else { return [] }
        let asset = AVURLAsset(url: url)
        let reader: AVAssetReader
        do { reader = try AVAssetReader(asset: asset) } catch { throw EG2VideoError.undecodable(error.localizedDescription) }
        guard let track = asset.tracks(withMediaType: .video).first else { throw EG2VideoError.noVideoTrack }
        // Ask for the decoder's own 8-bit 4:2:0 YCbCr planes and convert them ourselves
        // (see `yuvToRGB`): AVFoundation's BGRA conversion upsamples chroma smoothly and
        // differs from the reference decoder (ffmpeg/swscale) by ~3/255 on average.
        let colour = EG2YUVSpace(track: track)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: colour.fullRange
                ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ])
        out.alwaysCopiesSampleData = false
        guard reader.canAdd(out) else { throw EG2VideoError.undecodable("cannot read the video track") }
        reader.add(out)
        guard reader.startReading() else {
            throw EG2VideoError.undecodable(reader.error?.localizedDescription ?? "cannot start reading")
        }
        var wanted = indices[...]
        var result: [T] = []
        result.reserveCapacity(indices.count)
        var n = 0
        while n <= last, let sb = out.copyNextSampleBuffer() {
            guard let px = CMSampleBufferGetImageBuffer(sb) else { continue }
            // Frames past the end of a clip that reports fewer frames than it has are not
            // asked for; a requested index repeated (very low fps) reuses the same frame.
            while let want = wanted.first, want == n {
                result.append(try transform(try Self.yuvToRGB(px, colour)))
                wanted = wanted.dropFirst()
            }
            n += 1
        }
        if reader.status == .failed {
            throw EG2VideoError.undecodable(reader.error?.localizedDescription ?? "decode failed")
        }
        // The container promised more frames than it decoded.
        guard result.count == indices.count else {
            throw EG2VideoError.undecodable("decoded \(n) frames but frame index \(wanted.first ?? last) was requested")
        }
        return result
    }

    /// 8-bit 4:2:0 bi-planar YCbCr -> 8-bit RGB the way the reference decoder's
    /// swscale does for 4:2:0 H.264: nearest-neighbour chroma (each chroma sample
    /// covers its 2x2 luma block), the stream's matrix (BT.601 when untagged, which
    /// is swscale's default) and range. Measured on the fixtures against torchcodec
    /// frames: mean |diff| 0.19 / 255, max 1 (AVFoundation's own BGRA output: 2.9 mean).
    static func yuvToRGB(_ px: CVPixelBuffer, _ space: EG2YUVSpace) throws -> EG2RGBImage {
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let w = CVPixelBufferGetWidth(px), h = CVPixelBufferGetHeight(px)
        let fmt = CVPixelBufferGetPixelFormatType(px)
        guard fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
              CVPixelBufferGetPlaneCount(px) == 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(px, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(px, 1), w > 0, h > 0 else {
            throw EG2VideoError.undecodable("unexpected pixel format")
        }
        let full = fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        let yBpr = CVPixelBufferGetBytesPerRowOfPlane(px, 0), cBpr = CVPixelBufferGetBytesPerRowOfPlane(px, 1)
        let yp = yBase.assumingMemoryBound(to: UInt8.self), cp = cBase.assumingMemoryBound(to: UInt8.self)
        let kr = space.kr, kb = space.kb, kg = 1 - kr - kb
        let yScale: Float = full ? 1 : 255.0 / 219.0, cScale: Float = full ? 1 : 255.0 / 224.0
        let yOff: Float = full ? 0 : 16
        let crR = 2 * (1 - kr), cbB = 2 * (1 - kb)
        let gCr = kr * crR / kg, gCb = kb * cbB / kg
        var out = [UInt8](repeating: 0, count: w * h * 3)
        out.withUnsafeMutableBufferPointer { o in
            for y in 0 ..< h {
                let yRow = yp + y * yBpr
                let cRow = cp + (y / 2) * cBpr
                for x in 0 ..< w {
                    let yy = (Float(yRow[x]) - yOff) * yScale
                    let cb = (Float(cRow[(x / 2) * 2]) - 128) * cScale
                    let cr = (Float(cRow[(x / 2) * 2 + 1]) - 128) * cScale
                    let r = yy + crR * cr
                    let b = yy + cbB * cb
                    let g = yy - gCr * cr - gCb * cb
                    let k = (y * w + x) * 3
                    o[k] = clamp8(r); o[k + 1] = clamp8(g); o[k + 2] = clamp8(b)
                }
            }
        }
        return EG2RGBImage(pixels: out, width: w, height: h)
    }

    @inline(__always) private static func clamp8(_ v: Float) -> UInt8 {
        let r = (v + 0.5).rounded(.down)
        return r <= 0 ? 0 : (r >= 255 ? 255 : UInt8(r))
    }
}

/// YCbCr -> RGB parameters of a video track: the luma coefficients of its matrix
/// (BT.709 / BT.601 / BT.2020 / SMPTE 240M, from the track's colour tags) and
/// whether the range is full. An UNtagged stream is BT.601 limited range, which is
/// what the reference decoder (swscale) assumes whatever the frame size.
struct EG2YUVSpace {
    var kr: Float = 0.299, kb: Float = 0.114
    var fullRange = false

    init(track: AVAssetTrack) {
        guard let fd = track.formatDescriptions.first else { return }
        // swiftlint:disable:next force_cast
        let desc = fd as! CMFormatDescription
        if let m = CMFormatDescriptionGetExtension(desc, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String {
            switch m {
            case let s where s == (kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String): kr = 0.2126; kb = 0.0722
            case let s where s == (kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as String): kr = 0.2627; kb = 0.0593
            case let s where s == (kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995 as String): kr = 0.212; kb = 0.087
            default: break  // ITU_R_601_4 and anything unknown
            }
        }
        if let f = CMFormatDescriptionGetExtension(desc, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool {
            fullRange = f
        }
    }
}
