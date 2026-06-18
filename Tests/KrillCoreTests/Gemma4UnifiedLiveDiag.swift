import XCTest
import MLX
import Foundation
import KrillTokenizer
@testable import KrillCore

/// Live diagnostic against the real 12B checkpoint. Skipped unless
/// `KRILL_G4U_MODEL` points at the snapshot dir. Not a CI gate.
final class Gemma4UnifiedLiveDiag: XCTestCase {

    func testImagePlaceholderTokenizesToImageId() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL to the snapshot dir")
        }
        let tk = try await KrillTokenizer(from: URL(fileURLWithPath: path))
        let msgs = [["role": "user", "content": "<|image|><|image|><|image|>What color?"]]
        let ids = tk.formatGemma4TokenIds(messages: msgs)
        let count258880 = ids.filter { $0 == 258880 }.count
        print("formatGemma4TokenIds: 258880 count = \(count258880); total ids = \(ids.count)")
        print("first 25 ids: \(Array(ids.prefix(25)))")
        XCTAssertEqual(count258880, 3, "the 3 <|image|> placeholders must encode to id 258880")
    }

    /// Gate the LOAD-BEARING claim that the begin/end marker strings emitted by
    /// the serving path round-trip through the Gemma tokenizer to the exact ids
    /// the model was trained with. Env-gated (needs the real tokenizer); the
    /// pure prefix STRUCTURE is gated in CI by Gemma4UnifiedMarkersTests.
    func testMediaMarkerTokenIds() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL to the snapshot dir")
        }
        let tk = try await KrillTokenizer(from: URL(fileURLWithPath: path))
        // (marker text, expected single id)
        let cases: [(String, Int)] = [
            ("<|image|>", 258880), ("<|audio|>", 258881),
            ("<|image>", 255999),  ("<image|>", 258882),
            ("<|audio>", 256000),  ("<audio|>", 258883),
        ]
        for (text, id) in cases {
            let ids = tk.encode(text)
            print("encode(\(text)) = \(ids)")
            XCTAssertEqual(ids.filter { $0 == id }.count, 1,
                "\(text) must encode to a single id \(id)")
        }
    }

    func testVisionFeaturesDifferByColor() throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL to the snapshot dir")
        }
        let loaded = try loadModel(from: URL(fileURLWithPath: path))
        guard let model = loaded.module as? Gemma4UnifiedModel else {
            return XCTFail("not a Gemma4UnifiedModel: family=\(loaded.family)")
        }

        func featuresFor(_ pngPath: String) throws -> MLXArray {
            let data = try Data(contentsOf: URL(fileURLWithPath: pngPath))
            let packed = try preprocessGemma4UnifiedImage(data, modelPatchSize: model.visionConfig.modelPatchSize)
            let (pixels, pos) = Gemma4UnifiedModel.unpackImage(packed, patchDim: model.visionConfig.patchDim)
            let feats = model.encodeImage(pixels!, positionIds: pos)
            eval(feats)
            return feats
        }

        let red = try featuresFor("/tmp/red.png")
        let blue = try featuresFor("/tmp/blue.png")
        print("RED feats shape=\(red.shape) mean=\(MLX.mean(red).item(Float.self)) std=\(MLX.std(red).item(Float.self))")
        print("BLUE feats shape=\(blue.shape) mean=\(MLX.mean(blue).item(Float.self)) std=\(MLX.std(blue).item(Float.self))")
        let diff = MLX.mean(MLX.abs(red - blue)).item(Float.self)
        print("RED vs BLUE mean abs diff = \(diff)")

        // Also probe the raw patch input + the vision_embedder output before
        // the projection, to localize a constant-output bug.
        let redData = try Data(contentsOf: URL(fileURLWithPath: "/tmp/red.png"))
        let blueData = try Data(contentsOf: URL(fileURLWithPath: "/tmp/blue.png"))
        let rp = try preprocessGemma4UnifiedImage(redData, modelPatchSize: model.visionConfig.modelPatchSize)
        let bp = try preprocessGemma4UnifiedImage(blueData, modelPatchSize: model.visionConfig.modelPatchSize)
        let (rpix, rpos) = Gemma4UnifiedModel.unpackImage(rp, patchDim: model.visionConfig.patchDim)
        let (bpix, _) = Gemma4UnifiedModel.unpackImage(bp, patchDim: model.visionConfig.patchDim)
        print("RAW patch diff (red vs blue) = \(MLX.mean(MLX.abs(rpix! - bpix!)).item(Float.self))")
        print("RAW red patch[0,0,:6] = \(rpix![0, 0, 0..<6])")
        print("RAW blue patch[0,0,:6] = \(bpix![0, 0, 0..<6])")
        let rEmbed = model.visionEmbedder(rpix!, positionIds: rpos)
        let bEmbed = model.visionEmbedder(bpix!, positionIds: rpos)
        print("vision_embedder out diff = \(MLX.mean(MLX.abs(rEmbed - bEmbed)).item(Float.self))")

        XCTAssertGreaterThan(diff, 1e-3, "red and blue must produce different vision features")
    }

    /// Decisive: run a full multimodal forward with a real image-token prompt
    /// for red vs blue and compare next-token logits. Different logits => the
    /// image features genuinely reach and influence the decoder.
    func testFullForwardLogitsDifferByColor() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL to the snapshot dir")
        }
        let loaded = try loadModel(from: URL(fileURLWithPath: path))
        guard let model = loaded.module as? Gemma4UnifiedModel else { return XCTFail("wrong family") }
        let tk = try await KrillTokenizer(from: URL(fileURLWithPath: path))

        func argmaxNextToken(_ pngPath: String) throws -> (Int, MLXArray) {
            let data = try Data(contentsOf: URL(fileURLWithPath: pngPath))
            let packed = try preprocessGemma4UnifiedImage(data, modelPatchSize: model.visionConfig.modelPatchSize)
            let nTokens = packed.dim(1)
            let imgRun = String(repeating: "<|image|>", count: nTokens)
            let msgs = [["role": "user", "content": imgRun + "What is the dominant color? One word."]]
            let ids = tk.formatGemma4TokenIds(messages: msgs)
            let count = ids.filter { $0 == 258880 }.count
            XCTAssertEqual(count, nTokens, "prompt image-token count must match feature count")
            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let (pixels, pos) = Gemma4UnifiedModel.unpackImage(packed, patchDim: model.visionConfig.patchDim)
            let logits = model(tokens, caches: nil, pixelValues: pixels,
                               imagePositionIds: pos, audioFeatures: nil,
                               mediaHash: nil, lastTokenOnly: true)
            eval(logits)
            let last = logits[0, logits.dim(1) - 1]
            let am = MLX.argMax(last).item(Int.self)
            return (am, last)
        }

        let (redTok, redLogits) = try argmaxNextToken("/tmp/red.png")
        let (blueTok, blueLogits) = try argmaxNextToken("/tmp/blue.png")
        let logitDiff = MLX.mean(MLX.abs(redLogits - blueLogits)).item(Float.self)
        print("RED argmax next-token id = \(redTok) -> \(tk.decode([redTok]))")
        print("BLUE argmax next-token id = \(blueTok) -> \(tk.decode([blueTok]))")
        print("logit mean-abs diff red vs blue = \(logitDiff)")
        XCTAssertGreaterThan(logitDiff, 1e-2, "image features must influence the decoder logits")
    }

    /// Greedy-generate the actual answer directly through the model (no engine,
    /// no KV cache) for red vs blue, to see whether the model itself reads the
    /// color. Re-runs the full prompt each step (slow but only ~24 steps).
    func testDirectGreedyAnswer() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL to the snapshot dir")
        }
        let loaded = try loadModel(from: URL(fileURLWithPath: path))
        guard let model = loaded.module as? Gemma4UnifiedModel else { return XCTFail("wrong family") }
        let tk = try await KrillTokenizer(from: URL(fileURLWithPath: path))

        func generate(_ pngPath: String, steps: Int) throws -> String {
            let data = try Data(contentsOf: URL(fileURLWithPath: pngPath))
            let packed = try preprocessGemma4UnifiedImage(data, modelPatchSize: model.visionConfig.modelPatchSize)
            let nTokens = packed.dim(1)
            // Wrap the soft-token run with <start_of_image> (255999) and
            // <end_of_image> (258882), matching the reference processor's
            // `{boi}{image_token * n}{eoi}` expansion.
            let imgRun = "<|image>" + String(repeating: "<|image|>", count: nTokens) + "<image|>"
            let msgs = [["role": "user", "content": imgRun + "What is the dominant color of this image? Answer with just the color word."]]
            var ids = tk.formatGemma4TokenIds(messages: msgs)
            let (pixels, pos) = Gemma4UnifiedModel.unpackImage(packed, patchDim: model.visionConfig.patchDim)
            var generated: [Int] = []
            for _ in 0 ..< steps {
                let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
                let logits = model(tokens, caches: nil, pixelValues: pixels,
                                   imagePositionIds: pos, audioFeatures: nil,
                                   mediaHash: nil, lastTokenOnly: true)
                let last = logits[0, logits.dim(1) - 1]
                let next = MLX.argMax(last).item(Int.self)
                if next == 106 || next == 1 { break }   // <turn|> / eos
                ids.append(next)
                generated.append(next)
            }
            return tk.decode(generated)
        }

        let redOut = try generate("/tmp/red.png", steps: 24)
        let blueOut = try generate("/tmp/blue.png", steps: 24)
        print("RED  direct answer: \(redOut)")
        print("BLUE direct answer: \(blueOut)")
    }

    func testAudioFeatureStats() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_G4U_MODEL"] else {
            throw XCTSkip("set KRILL_G4U_MODEL")
        }
        let loaded = try loadModel(from: URL(fileURLWithPath: path))
        guard let model = loaded.module as? Gemma4UnifiedModel else { return XCTFail("wrong family") }
        let wavData = try Data(contentsOf: URL(fileURLWithPath: "/tmp/tone.wav"))
        let wave = try AudioPreprocessor.monoWaveform(fromAudio: wavData)
        print("waveform samples=\(wave.count) min=\(wave.min() ?? 0) max=\(wave.max() ?? 0)")
        let frames = preprocessGemma4UnifiedAudio(wave)
        print("frames shape=\(frames.shape)")
        let feats = model.encodeAudio(frames)
        eval(feats)
        let m = MLX.mean(feats).item(Float.self)
        let sd = MLX.std(feats).item(Float.self)
        let mx = MLX.max(MLX.abs(feats)).item(Float.self)
        print("audio feats shape=\(feats.shape) mean=\(m) std=\(sd) maxabs=\(mx)")
        XCTAssertFalse(m.isNaN || sd.isNaN, "audio features must not be NaN")
    }
}
