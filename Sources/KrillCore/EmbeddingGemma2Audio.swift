import Foundation
import MLX
import MLXNN
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

// MARK: - EmbeddingGemma 2: audio
//
// The checkpoint's `audio_tower.*` / `embed_audio.*` tensors are the Gemma 4
// USM conformer (`gemma4_audio`: 12 layers, d 1024, 8 heads, chunked local
// attention, clipped linears ON, `output_proj` 1024 -> 1536 WITH bias), so
// Krill's existing `AudioEncoder` + `MultimodalEmbedder` run it as is (keys
// verified key-for-key by the strict loader below). This file adds only what is
// specific to embeddings:
//
//   * `EG2AudioPreprocessor`: decode to mono 16 kHz float32 (AVFoundation,
//     high-quality resample), the HF `Gemma4AudioFeatureExtractor` features
//     (shared with the Gemma 4 chat path: `AudioPreprocessor`), and the
//     soft-token count of `EmbeddingGemma2Processor.replace_audio_token`
//     (mask simulation through the two stride-2 convs). NOT capped at 280: the
//     reference only truncates the waveform at 30 s (750 tokens); the
//     `audio_seq_length` cap lives in a serving helper the reference never calls.
//   * `EG2AudioTower` (`audio_tower` + `embed_audio`, keys equal the
//     checkpoint's) and `loadEG2AudioTower` (strict binding).

// MARK: Errors

public enum EG2AudioError: Error, CustomStringConvertible, Equatable {
    case undecodable(String)
    case tooShort(samples: Int)
    case tooLong(seconds: Double, maxSeconds: Double)
    case platformUnavailable

    public var description: String {
        switch self {
        case .undecodable(let why):
            return "audio could not be decoded (\(why)); supported: wav, mp3, m4a/aac, flac, aiff, caf"
        case .tooShort(let n):
            return "audio is too short (\(n) samples at 16 kHz); the minimum is \(EG2AudioPreprocessor.minSamples) samples (about 0.1 s)"
        case .tooLong(let s, let m):
            return String(format: "audio is %.1f s; the maximum is %.0f s per clip (the model's feature extractor truncates beyond it, Krill refuses instead)", s, m)
        case .platformUnavailable:
            return "audio decoding is not available on this platform"
        }
    }
}

// MARK: Preprocessor

/// Features for one clip.
public struct EG2PreparedAudio {
    /// Log-mel `[1, T, 128]` float32 and validity `[1, T]` (true = real audio).
    public let mel: MLXArray
    public let validMask: MLXArray
    public let numSamples: Int
    /// Number of `<audio>` placeholders for this clip.
    public let softTokens: Int
    public var seconds: Double { Double(numSamples) / Double(EG2AudioPreprocessor.sampleRate) }
}

public enum EG2AudioPreprocessor {
    public static let sampleRate = 16_000
    /// `Gemma4AudioFeatureExtractor.__call__(max_length=480000)`.
    public static let maxSamples = 480_000
    public static var maxSeconds: Double { Double(maxSamples) / Double(sampleRate) }
    /// Below this the clip has no valid conv frame (0 soft tokens).
    public static let minSamples = 1_600
    static let frameLength = 320, hop = 160, padMultiple = 128

    /// Mel frames the extractor produces for `n` samples (after padding to a
    /// multiple of 128 and the semicausal left pad of frame_length / 2).
    static func melFrames(samples n: Int) -> Int {
        let padded = n % padMultiple == 0 ? n : (n / padMultiple + 1) * padMultiple
        let L = frameLength / 2 + padded
        return L >= frameLength + 1 ? (L - (frameLength + 1)) / hop + 1 : 0
    }

    /// Soft tokens `EmbeddingGemma2Processor.replace_audio_token` inserts: the
    /// per-frame validity mask (frame i is valid iff its last sample
    /// `i*hop + frame_length` is a real sample, i.e. `i*hop + frame_length <
    /// frame_length/2 + n`) sub-sampled twice with `mask[::2][:t_out]`. Two
    /// stride-2 steps keep every 4th frame. NOT capped at 280.
    public static func softTokenCount(samples n: Int) -> Int {
        let T = melFrames(samples: n)
        guard T > 0 else { return 0 }
        var count = 0
        var i = 0
        while i < T {
            if i * hop + frameLength < frameLength / 2 + n { count += 1 }
            i += 4
        }
        return count
    }

    /// Validate length and compute the soft-token count (cheap, no mel work), so a
    /// caller can size the prompt and reject an over-long one before any tower run.
    public static func softTokens(forWaveform w: [Float]) throws -> Int {
        if w.count < minSamples { throw EG2AudioError.tooShort(samples: w.count) }
        if w.count > maxSamples {
            throw EG2AudioError.tooLong(seconds: Double(w.count) / Double(sampleRate), maxSeconds: maxSeconds)
        }
        let n = softTokenCount(samples: w.count)
        if n == 0 { throw EG2AudioError.tooShort(samples: w.count) }
        return n
    }

    /// Features for a mono 16 kHz waveform.
    public static func prepare(waveform w: [Float]) throws -> EG2PreparedAudio {
        let n = try softTokens(forWaveform: w)
        let f = try AudioPreprocessor.features(waveform: w)
        return EG2PreparedAudio(mel: f.mel, validMask: f.validMask, numSamples: w.count, softTokens: n)
    }

    // MARK: Decode

    /// Decode any AVFoundation-readable audio (WAV, MP3, M4A/AAC, FLAC, AIFF,
    /// CAF, ...) to mono float32 at 16 kHz. Channels are averaged; a sample rate
    /// other than 16 kHz goes through `AVAudioConverter` (anti-aliased).
    /// `formatHint` (e.g. "wav", "mp3") only picks the temp file extension.
    public static func decode(_ data: Data, formatHint: String? = nil) throws -> [Float] {
        #if canImport(AVFoundation)
        guard !data.isEmpty else { throw EG2AudioError.undecodable("empty") }
        let ext = sanitizedExtension(formatHint) ?? sniffExtension(data)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-eg2-audio-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("input.\(ext)")
        try data.write(to: url, options: .atomic)

        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false) }
        catch { throw EG2AudioError.undecodable("\(error.localizedDescription)") }
        let fmt = file.processingFormat
        let frames = Int(file.length)
        guard frames > 0, fmt.channelCount > 0 else { throw EG2AudioError.undecodable("no audio frames") }
        // Refuse absurd lengths before allocating (30 s + slack at the source rate).
        let srcRate = fmt.sampleRate
        if Double(frames) / srcRate > maxSeconds * 4 + 5 {
            throw EG2AudioError.tooLong(seconds: Double(frames) / srcRate, maxSeconds: maxSeconds)
        }
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frames)) else {
            throw EG2AudioError.undecodable("buffer allocation failed")
        }
        do { try file.read(into: buf) } catch { throw EG2AudioError.undecodable("\(error.localizedDescription)") }
        let n = Int(buf.frameLength)
        guard n > 0, let ch = buf.floatChannelData else { throw EG2AudioError.undecodable("no samples") }
        let nch = Int(fmt.channelCount)
        var mono = [Float](repeating: 0, count: n)
        if nch == 1 {
            mono = Array(UnsafeBufferPointer(start: ch[0], count: n))
        } else {
            for c in 0 ..< nch { for i in 0 ..< n { mono[i] += ch[c][i] } }
            let inv = 1 / Float(nch)
            for i in 0 ..< n { mono[i] *= inv }
        }
        // AVAudioFile does not honour the MP3 encoder delay / padding that a LAME or
        // Xing "Info" tag records; ffmpeg (the reference decoder) does. Trim the same
        // samples so clip length and soft-token count match.
        if ext == "mp3", let t = mp3GaplessTrim(data), t.start + t.end < mono.count {
            mono = Array(mono[t.start ..< (mono.count - t.end)])
        }
        if Int(srcRate.rounded()) == sampleRate { return mono }
        return try resample(mono, from: srcRate)
        #else
        throw EG2AudioError.platformUnavailable
        #endif
    }

    #if canImport(AVFoundation)
    private static func resample(_ mono: [Float], from srcRate: Double) throws -> [Float] {
        guard let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: srcRate,
                                        channels: 1, interleaved: false),
              let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
                                         channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inFmt, to: outFmt),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(mono.count))
        else { throw EG2AudioError.undecodable("resampler unavailable for \(Int(srcRate)) Hz") }
        inBuf.frameLength = AVAudioFrameCount(mono.count)
        mono.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: mono.count) }
        let cap = AVAudioFrameCount(Double(mono.count) * Double(sampleRate) / srcRate) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: cap) else {
            throw EG2AudioError.undecodable("buffer allocation failed")
        }
        var fed = false
        var err: NSError?
        let status = conv.convert(to: outBuf, error: &err) { _, st in
            if fed { st.pointee = .endOfStream; return nil }
            fed = true; st.pointee = .haveData; return inBuf
        }
        if status == .error || outBuf.frameLength == 0 {
            throw EG2AudioError.undecodable("resampling failed: \(err?.localizedDescription ?? "unknown")")
        }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    /// Samples to drop from the start / end of an MP3's decoded output, from the
    /// encoder delay and padding in its first frame's Xing/Info (LAME) tag, the way
    /// ffmpeg does. AVAudioFile already removes the 529-sample decoder delay, so what
    /// is left to drop is exactly the tag's `delay` at the start and `padding` at the end
    /// (measured: AVFoundation output is ffmpeg output shifted by `delay`). nil when
    /// the file has no such tag (then nothing is trimmed, as in ffmpeg).
    static func mp3GaplessTrim(_ d: Data) -> (start: Int, end: Int)? {
        let b = [UInt8](d.prefix(4096))
        var o = 0
        if b.count >= 10, b[0] == 0x49, b[1] == 0x44, b[2] == 0x33 {  // ID3v2: skip it
            o = 10 + (Int(b[6] & 0x7F) << 21 | Int(b[7] & 0x7F) << 14 | Int(b[8] & 0x7F) << 7 | Int(b[9] & 0x7F))
        }
        // Resync to the first frame header (Layer III).
        while o + 4 < b.count, !(b[o] == 0xFF && (b[o + 1] & 0xE0) == 0xE0 && (b[o + 1] >> 1) & 3 == 1) { o += 1 }
        guard o + 4 < b.count else { return nil }
        let version = (b[o + 1] >> 3) & 3            // 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
        let mono = (b[o + 3] >> 6) & 3 == 3
        let crc = b[o + 1] & 1 == 0
        let side = version == 3 ? (mono ? 17 : 32) : (mono ? 9 : 17)
        var p = o + 4 + (crc ? 2 : 0) + side
        guard p + 8 < b.count, ["Xing", "Info"].contains(String(decoding: b[p ..< p + 4], as: UTF8.self)) else { return nil }
        let flags = Int(b[p + 7])
        p += 8
        if flags & 1 != 0 { p += 4 }
        if flags & 2 != 0 { p += 4 }
        if flags & 4 != 0 { p += 100 }
        if flags & 8 != 0 { p += 4 }
        guard p + 24 <= b.count, b[p] != 0 else { return nil }  // encoder string present
        let delay = Int(b[p + 21]) << 4 | Int(b[p + 22]) >> 4
        let padding = Int(b[p + 22] & 0x0F) << 8 | Int(b[p + 23])
        return (delay, padding)
    }

    static func sanitizedExtension(_ hint: String?) -> String? {
        guard var h = hint?.lowercased().trimmingCharacters(in: .whitespaces), !h.isEmpty else { return nil }
        if let slash = h.lastIndex(of: "/") { h = String(h[h.index(after: slash)...]) }  // "audio/mpeg"
        switch h {
        case "mpeg", "mpga", "mp3": return "mp3"
        case "x-wav", "wave", "wav": return "wav"
        case "x-m4a", "mp4", "aac", "m4a", "x-aac": return "m4a"
        case "x-flac", "flac": return "flac"
        case "aif", "aiff", "x-aiff": return "aiff"
        case "caf", "x-caf": return "caf"
        default:
            return h.allSatisfy({ $0.isLetter || $0.isNumber }) && h.count <= 5 ? h : nil
        }
    }

    static func sniffExtension(_ d: Data) -> String {
        let b = [UInt8](d.prefix(12))
        func tag(_ o: Int, _ s: String) -> Bool { b.count >= o + s.utf8.count && Array(b[o ..< o + s.utf8.count]) == Array(s.utf8) }
        if tag(0, "RIFF") && tag(8, "WAVE") { return "wav" }
        if tag(0, "fLaC") { return "flac" }
        if tag(0, "FORM") { return "aiff" }
        if tag(0, "caff") { return "caf" }
        if tag(0, "ID3") || (b.count >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0) { return "mp3" }
        if tag(4, "ftyp") { return "m4a" }
        return "audio"
    }
    #endif
}

// MARK: Tower

/// `audio_tower.*` + `embed_audio.*`. Parameter keys equal the checkpoint's keys.
public final class EG2AudioTower: Module {
    @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder
    public private(set) var computeDtype: DType = .float32

    init(_ cfg: AudioConfig, textHidden: Int) {
        _audioTower = ModuleInfo(wrappedValue: AudioEncoder(cfg), key: "audio_tower")
        _embedAudio = ModuleInfo(wrappedValue: MultimodalEmbedder(
            embeddingDim: cfg.outputProjDims, textHiddenSize: textHidden, eps: cfg.rmsNormEps), key: "embed_audio")
    }

    public func setComputeDtype(_ dtype: DType) {
        update(parameters: parameters().mapValues { $0.asType(dtype) })
        computeDtype = dtype
    }

    /// One clip -> soft tokens in text space `[softTokens, textHidden]`: the
    /// conformer, then ONLY the valid frames (the reference indexes the encoder
    /// output with its own mask), then scale-free RMSNorm + 1536 -> 512.
    public func softTokens(_ a: EG2PreparedAudio) throws -> MLXArray {
        let mel = a.mel.asType(computeDtype)
        let (enc, invalid) = audioTower(mel, validMask: a.validMask)  // [1, T', 1536], [1, T']
        // Valid frames form a prefix (right padding only); count them from the tower's own mask.
        let valid = (invalid .== MLXArray(false)).asType(.int32).sum().item(Int.self)
        guard valid == a.softTokens else {
            throw EG2SequenceError.softTokenMismatch(modality: "audio", expected: a.softTokens, got: valid)
        }
        let soft = embedAudio(enc[0..., 0 ..< valid, 0...])
        return soft.reshaped(valid, soft.dim(2))
    }
}

// MARK: Loader (strict binding)

public struct EG2AudioLoadReport: Sendable, CustomStringConvertible {
    public let bound: Int
    public var description: String {
        "EmbeddingGemma2 audio: bound \(bound) tensors from the checkpoint (strict: every one consumed, none missing)"
    }
}

public enum EG2AudioLoadError: Error, CustomStringConvertible {
    case noAudioTower
    case unsupported(String)
    public var description: String {
        switch self {
        case .noAudioTower: return "this checkpoint has no audio tower (no audio_tower.* tensors)"
        case .unsupported(let m): return "unsupported audio tower: \(m)"
        }
    }
}

/// The HF checkpoint stores the two conv kinds in PyTorch layout, which is NOT
/// the channel-last layout the Gemma 4 / mlx-vlm checkpoints (and Krill's
/// `AudioEncoder`) use: `subsample_conv_projection.*.conv.weight` is
/// `[out, in, kH, kW]` (module wants `[out, kH, kW, in]`) and
/// `lconv1d.depthwise_conv1d.weight` is `[C, 1, K]` (module wants `[C, K, 1]`).
/// Convert only when the transposed shape is exactly the module's shape (an
/// already channel-last tensor passes through; anything else is left for the
/// strict shape check to reject).
func eg2AudioConvToMLXLayout(key: String, value: MLXArray, expected: [Int]?) -> MLXArray {
    guard let expected, value.shape != expected else { return value }
    if key.hasSuffix(".conv.weight"), value.ndim == 4 {
        let t = value.transposed(0, 2, 3, 1)
        if t.shape == expected { return t }
    } else if key.hasSuffix("depthwise_conv1d.weight"), value.ndim == 3 {
        let t = value.transposed(0, 2, 1)
        if t.shape == expected { return t }
    }
    return value
}

/// Load the audio tower + `embed_audio` with STRICT binding: every
/// `audio_tower.*` / `embed_audio.*` tensor must land on a parameter and every
/// parameter must be covered with the right shape (clip scalars included:
/// `use_clipped_linears` is true here, so the checkpoint carries them).
public func loadEG2AudioTower(
    directory: URL, weights: [String: MLXArray]? = nil, dtype: DType = .float32
) throws -> (tower: EG2AudioTower, report: EG2AudioLoadReport) {
    let cfgData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    let root = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any]
    guard let acfg = root?["audio_config"] as? [String: Any] else { throw EG2AudioLoadError.noAudioTower }
    if let clipped = acfg["use_clipped_linears"] as? Bool, !clipped {
        throw EG2AudioLoadError.unsupported("use_clipped_linears=false (the checkpoint's audio tower is clipped)")
    }
    let tcfg = try JSONDecoder().decode(EmbeddingGemma2Config.self, from: cfgData)
    let all = try weights ?? loadWeightArrays(from: directory)
    let audio = all.filter { $0.key.hasPrefix("audio_tower.") || $0.key.hasPrefix("embed_audio.") }
    guard audio.keys.contains(where: { $0.hasPrefix("audio_tower.") }) else { throw EG2AudioLoadError.noAudioTower }
    let tower = EG2AudioTower(AudioConfig(from: acfg), textHidden: tcfg.hiddenSize)
    let expected = Dictionary(uniqueKeysWithValues: tower.parameters().flattened().map { ($0.0, $0.1.shape) })
    let bound = audio.map { (k, v) in (k, eg2AudioConvToMLXLayout(key: k, value: v, expected: expected[k])) }
    try tower.update(
        parameters: ModuleParameters.unflattened(bound),
        verify: [.allModelKeysSet, .shapeMismatch, .noUnusedKeys])
    tower.setComputeDtype(dtype)
    eval(tower)
    let report = EG2AudioLoadReport(bound: audio.count)
    FileHandle.standardError.write(Data((report.description + "\n").utf8))
    return (tower, report)
}
