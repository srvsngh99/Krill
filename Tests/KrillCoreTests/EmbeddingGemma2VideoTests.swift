import XCTest
import Foundation
@testable import KrillCore

/// EmbeddingGemma 2 video path without the real checkpoint: the frame sampler
/// (values computed with the reference's own `sample_frames` + numpy, see
/// docs/EMBEDDINGGEMMA2.md), container metadata, frame decoding with
/// AVFoundation, and the per-frame soft-token budget.
final class EmbeddingGemma2VideoTests: XCTestCase {

    private static func fixture(_ rel: String, _ file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("KrillEngineTests/Fixtures/eg2_mm/\(rel)")
    }

    // MARK: Sampler

    func testFrameIndicesMatchTheReferenceSampler() {
        // (total frames, native fps, duration s, indices): from the HF `sample_frames` + np.linspace.
        let cases: [(Int, Double, Double, [Int])] = [
            (30, 10.0, 3.0, [0, 10, 20]),
            (400, 10.0, 40.0, [0, 10, 20, 30, 50, 60, 70, 80, 100, 110, 120, 130, 150, 160, 170, 180, 200, 210, 220, 230,
                               250, 260, 270, 280, 300, 310, 320, 330, 350, 360, 370, 390]),
            (120, 30000.0 / 1001.0, 4.004, [0, 29, 59, 89]),
            (240, 24.0, 10.0, [0, 24, 48, 72, 96, 120, 144, 168, 192, 216]),
            (3600, 30.0, 120.0, [0, 90, 210, 330, 450, 570, 690, 780, 900, 1020, 1140, 1260, 1380, 1470, 1590, 1710, 1830,
                                 1950, 2070, 2160, 2280, 2400, 2520, 2640, 2760, 2850, 2970, 3090, 3210, 3330, 3450, 3570]),
            (50, 25.0, 2.0, [0, 25]),
            (1, 25.0, 0.04, [0]),
            (90, 59.94, 1.5, [0]),
            (100, 0.5, 200.0, [0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 32, 35, 38, 41, 44, 48, 51, 54, 57, 60, 64, 67, 70, 73,
                               77, 80, 83, 86, 89, 93, 96, 99]),
            (31, 1.0, 31.0, Array(0 ... 30)),
            (32, 1.0, 32.0, Array(0 ... 31)),
            (33, 1.0, 33.0, Array(0 ... 30) + [32]),
            (7200, 24000.0 / 1001.0, 300.3, [0, 215, 455, 671, 911, 1150, 1366, 1606, 1846, 2061, 2301, 2541, 2757, 2997,
                                             3236, 3452, 3692, 3908, 4147, 4387, 4603, 4843, 5082, 5298, 5538, 5778, 5994,
                                             6233, 6473, 6689, 6929, 7168]),
            (10, 3.0, 3.3, [0, 3, 6]),
        ]
        for (total, fps, dur, want) in cases {
            XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: total, fps: fps, duration: dur), want,
                           "total \(total) fps \(fps) duration \(dur)")
        }
    }

    func testSamplerInvariants() {
        for (total, fps, dur) in [(30, 10.0, 3.0), (400, 10.0, 40.0), (97, 29.97, 3.23), (5000, 25.0, 200.0), (2, 1.0, 1.9)] {
            let idx = EG2VideoSampler.frameIndices(totalFrames: total, fps: fps, duration: dur)
            XCTAssertLessThanOrEqual(idx.count, 32)
            XCTAssertFalse(idx.isEmpty)
            XCTAssertEqual(idx, idx.sorted(), "ascending")
            XCTAssertTrue(idx.allSatisfy { $0 >= 0 && $0 < total })
        }
        // A sub-second clip still yields one frame (max(1, int(duration))).
        XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: 12, fps: 24, duration: 0.5), [0])
        // Degenerate metadata yields nothing rather than a crash.
        XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: 0, fps: 24, duration: 3), [])
        XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: 10, fps: 0, duration: 3), [])
        XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: 10, fps: 24, duration: 0), [])
    }

    func testPerFrameTokenBudgetMatchesTheReference() throws {
        // 320x240 and 480x270 frames at the 140-token video budget: 130 / 120 soft tokens (reference).
        XCTAssertEqual(try EG2ImagePreprocessor.softTokens(width: 320, height: 240, maxSoftTokens: 140), 130)
        XCTAssertEqual(try EG2ImagePreprocessor.softTokens(width: 480, height: 270, maxSoftTokens: 140), 120)
        XCTAssertEqual(try EG2ImagePreprocessor.softTokens(width: 256, height: 192, maxSoftTokens: 140), 130)
        // and the image budget is untouched
        XCTAssertEqual(try EG2ImagePreprocessor.softTokens(width: 640, height: 480), 266)
    }

    // MARK: Container + decode (AVFoundation)

    func testInspectFixtures() throws {
        let v3 = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v3.mp4"))).info
        XCTAssertEqual(v3.totalFrames, 30)
        XCTAssertEqual(v3.fps, 10, accuracy: 0.01)
        XCTAssertEqual(v3.duration, 3.0, accuracy: 0.01)
        XCTAssertEqual(v3.width, 320); XCTAssertEqual(v3.height, 240)

        let v40 = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v40.mp4"))).info
        XCTAssertEqual(v40.totalFrames, 400)
        XCTAssertEqual(v40.duration, 40.0, accuracy: 0.01)

        let ntsc = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v_ntsc.mp4"))).info
        XCTAssertEqual(ntsc.totalFrames, 120)
        XCTAssertEqual(ntsc.fps, 29.97, accuracy: 0.01)
        XCTAssertEqual(ntsc.duration, 4.004, accuracy: 0.01)
        XCTAssertEqual(EG2VideoSampler.frameIndices(totalFrames: ntsc.totalFrames, fps: ntsc.fps, duration: ntsc.duration),
                       [0, 29, 59, 89])
    }

    func testSampledIndicesForFixtures() throws {
        for (file, count, first, last) in [("v3.mp4", 3, 0, 20), ("v40.mp4", 32, 0, 390)] {
            let i = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/\(file)"))).info
            let idx = EG2VideoSampler.frameIndices(totalFrames: i.totalFrames, fps: i.fps, duration: i.duration)
            XCTAssertEqual(idx.count, count, file)
            XCTAssertEqual(idx.first, first, file)
            XCTAssertEqual(idx.last, last, file)
        }
    }

    func testDecodesExactlyTheRequestedFramesInOrder() throws {
        let src = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v3.mp4")))
        let frames = try src.frames(at: [0, 10, 20]) { $0 }
        XCTAssertEqual(frames.count, 3)
        for f in frames {
            XCTAssertEqual(f.width, 320); XCTAssertEqual(f.height, 240)
            XCTAssertEqual(f.pixels.count, 320 * 240 * 3)
        }
        // testsrc2 moves: different frames differ, the same index twice is identical.
        XCTAssertNotEqual(frames[0].pixels, frames[1].pixels)
        let again = try src.frames(at: [10, 10]) { $0 }
        XCTAssertEqual(again[0].pixels, frames[1].pixels)
        XCTAssertEqual(again[1].pixels, frames[1].pixels)
        // transform runs per frame and only its result is kept
        let means = try src.frames(at: [0, 29]) { f in f.pixels.reduce(0) { $0 + Int($1) } / f.pixels.count }
        XCTAssertEqual(means.count, 2)
    }

    /// The decoded pixels equal the reference decoder's (torchcodec / ffmpeg): each frame is
    /// reduced to 12x16 block means (`reference_video_frames.json`) and compared. With
    /// AVFoundation's own BGRA output this measured 2-3 / 255 mean and cost ~0.005 cosine;
    /// the swscale-style YUV conversion measures well under 0.5.
    func testDecodedFramesMatchTheReferenceDecoder() throws {
        struct Ref: Decodable { struct F: Decodable { let file: String; let index: Int; let w: Int; let h: Int; let thumb: [Double] }
                                let frames: [F] }
        let ref = try JSONDecoder().decode(Ref.self, from: Data(contentsOf: Self.fixture("reference_video_frames.json")))
        var worst = 0.0
        for file in Set(ref.frames.map { $0.file }) {
            let frames = ref.frames.filter { $0.file == file }
            let src = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/\(file)")))
            let got = try src.frames(at: frames.map { $0.index }) { $0 }
            for (r, g) in zip(frames, got) {
                XCTAssertEqual(g.width, r.w, file); XCTAssertEqual(g.height, r.h, file)
                let ye = (0 ... 12).map { $0 * g.height / 12 }, xe = (0 ... 16).map { $0 * g.width / 16 }
                var sum = 0.0
                for a in 0 ..< 12 {
                    for b in 0 ..< 16 {
                        for c in 0 ..< 3 {
                            var s = 0.0
                            for y in ye[a] ..< ye[a + 1] { for x in xe[b] ..< xe[b + 1] { s += Double(g.pixels[(y * g.width + x) * 3 + c]) } }
                            let mean = s / Double((ye[a + 1] - ye[a]) * (xe[b + 1] - xe[b]))
                            sum += abs(mean - r.thumb[(a * 16 + b) * 3 + c])
                        }
                    }
                }
                let mad = sum / 576
                worst = max(worst, mad)
                XCTAssertLessThan(mad, 0.5, "\(file) frame \(r.index): block-mean |diff| \(mad)")
            }
        }
        print("EG2 video frame fidelity vs torchcodec: worst block-mean abs diff \(worst) / 255")
    }

    func testVideoWithAnAudioTrackIsReadAsVideoOnly() throws {
        let src = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v_audio.mp4")))
        XCTAssertEqual(src.info.width, 256); XCTAssertEqual(src.info.height, 192)
        XCTAssertEqual(src.info.fps, 12, accuracy: 0.01)
        let idx = EG2VideoSampler.frameIndices(totalFrames: src.info.totalFrames, fps: src.info.fps, duration: src.info.duration)
        XCTAssertEqual(idx.count, 2)
        XCTAssertEqual(try src.frames(at: idx) { $0 }.count, 2)
    }

    func testGarbageAndAudioOnlyFilesAreClientErrors() throws {
        XCTAssertThrowsError(try EG2VideoSource(data: Data()))
        XCTAssertThrowsError(try EG2VideoSource(data: Data("definitely not a video".utf8))) {
            guard case EG2VideoError.undecodable = $0 else { return XCTFail("\($0)") }
        }
        // a PNG is not a video
        XCTAssertThrowsError(try EG2VideoSource(data: Data(contentsOf: Self.fixture("images/img_square_224.png"))))
    }

    func testTempFileIsRemovedWhenTheSourceGoesAway() throws {
        let tmp = FileManager.default.temporaryDirectory
        let before = try FileManager.default.contentsOfDirectory(atPath: tmp.path).filter { $0.hasPrefix("krill-eg2-video-") }
        do {
            let src = try EG2VideoSource(data: Data(contentsOf: Self.fixture("video/v3.mp4")))
            _ = try src.frames(at: [0]) { $0 }
        }
        let after = try FileManager.default.contentsOfDirectory(atPath: tmp.path).filter { $0.hasPrefix("krill-eg2-video-") }
        XCTAssertEqual(Set(after).subtracting(before), [])
    }

    func testFileExtensionSniffing() {
        let mp4 = Data([0, 0, 0, 0x18] + Array("ftypmp42".utf8))
        let mov = Data([0, 0, 0, 0x14] + Array("ftypqt  ".utf8))
        XCTAssertEqual(EG2VideoSource.fileExtension(hint: nil, data: mp4), "mp4")
        XCTAssertEqual(EG2VideoSource.fileExtension(hint: "", data: mov), "mov")
        XCTAssertEqual(EG2VideoSource.fileExtension(hint: "quicktime", data: mp4), "mov")
        XCTAssertEqual(EG2VideoSource.fileExtension(hint: "../../x", data: mp4), "mp4", "a hint can never choose a path")
    }
}
