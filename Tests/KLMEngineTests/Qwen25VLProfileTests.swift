import XCTest
import Foundation
import MLX
import KLMCache
@testable import KLMCore
@testable import KLMEngine

/// WS5 perf: per-stage profiler for the native Qwen 2.5-VL runtime.
///
/// The native forward is one lazy MLX graph that is `eval`'d once, so
/// Swift-level timers around `model(...)` measure graph-BUILD, not
/// execution. This profiler drives the model sub-component by
/// sub-component and puts an explicit `MLX.eval()` barrier after each
/// stage, so every printed number is real GPU execution time. It is
/// the measurement tool for the perf work: run it before and after
/// each optimization to see where the time actually goes.
///
/// Gated on `KLM_QWEN25VL_MODEL_PATH` (same gate as
/// `Qwen25VLSmokeTests`); skipped when unset. This is a profiler, not
/// a pass/fail correctness test - it asserts only that every stage
/// produced finite output and prints a timing table.
final class Qwen25VLProfileTests: XCTestCase {

    // MARK: - Gating

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo
            .environment["KLM_QWEN25VL_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    // MARK: - Timing helper

    /// Run `body` `warmup + iters` times, each followed by an
    /// `MLX.eval()` barrier on the value it returns, and report the
    /// median wall time of the measured runs in milliseconds.
    @discardableResult
    private func timeStage(
        _ label: String, warmup: Int = 3, iters: Int = 20,
        _ body: () -> MLXArray
    ) -> Double {
        for _ in 0 ..< warmup {
            let v = body()
            MLX.eval(v)
        }
        var samples: [Double] = []
        samples.reserveCapacity(iters)
        for _ in 0 ..< iters {
            let t0 = CFAbsoluteTimeGetCurrent()
            let v = body()
            MLX.eval(v)
            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
        }
        samples.sort()
        let median = samples[samples.count / 2]
        let lo = samples.first ?? 0
        let hi = samples.last ?? 0
        print(String(
            format: "  %-22@ median %8.3f ms   (min %7.3f, max %7.3f)",
            label as NSString, median, lo, hi))
        return median
    }

    // MARK: - Profiler

    func testProfileImagePromptStages() throws {
        let dir = try requireModel()
        let loaded = try loadModel(from: dir)
        guard let model = loaded.module
            as? Qwen25VLForConditionalGeneration else {
            return XCTFail("checkpoint did not load as a native VL model")
        }
        let cfg = model.config
        let vision = cfg.vision

        // -- Build a representative 224x224 image input --
        // A solid mid-gray tensor: pixel values do not affect timing,
        // only shapes do. 224x224 -> 16x16 patch grid -> 256 patches
        // -> 8x8 merged grid -> 64 <|image_pad|> tokens, matching the
        // headline benchmark image.
        let pixels = MLXArray.ones([224, 224, 3]).asType(.float32) * 0.5
        let patchBatch = Qwen25VLImagePreprocessor.toConv3DInput(
            Qwen25VLImagePreprocessor.normalize(pixels),
            patchSize: vision.patchSize,
            temporalPatchSize: vision.temporalPatchSize,
            spatialMergeSize: vision.spatialMergeSize)
        let mergeFactor = vision.patchSize * vision.spatialMergeSize
        let gridHMerged = 224 / mergeFactor
        let gridWMerged = 224 / mergeFactor
        let nImageTokens = gridHMerged * gridWMerged

        // -- Build a representative prompt: text + image span + text,
        //    totalling ~88 tokens (matches the benchmark prompt). --
        let imgPad = Int32(cfg.imageTokenId)
        let leadText: [Int32] = Array(1 ... 12)
        let tailText: [Int32] = Array(20 ... 31)
        let prompt32 = leadText
            + Array(repeating: imgPad, count: nImageTokens)
            + tailText
        let promptArray = MLXArray(prompt32).reshaped(1, prompt32.count)
        let imagePadStart = leadText.count

        print("\n=== Qwen 2.5-VL per-stage profile ===")
        print("  image grid (merged) : \(gridHMerged)x\(gridWMerged)"
            + "  image tokens: \(nImageTokens)")
        print("  prompt length       : \(prompt32.count) tokens")
        print("  vision depth        : \(vision.depth) blocks")

        // -- Stage 1: patch embed --
        timeStage("patchEmbed") {
            model.visual.patchEmbed(patchBatch)
        }

        // -- Stage 2: the vision transformer blocks --
        // Time both the mask-based windowed-attention path (the
        // pre-PR-58 implementation) and the batched-per-window
        // path the tower now uses in production. Showing both
        // keeps the delta auditable in CI: the batched stage
        // should be materially faster on a uniform grid.
        let embedded = model.visual.patchEmbed(patchBatch)
            .expandedDimensions(axis: 0)
        MLX.eval(embedded)
        let gridHFull = gridHMerged * vision.spatialMergeSize
        let gridWFull = gridWMerged * vision.spatialMergeSize
        let windowMask = Qwen25VLVisionTower.windowAttentionMask(
            gridHFull: gridHFull, gridWFull: gridWFull,
            vision: vision, dtype: embedded.dtype)
        let plan = Qwen25VLVisionTower.windowedAttentionPlan(
            gridHFull: gridHFull, gridWFull: gridWFull,
            vision: vision)
        timeStage("vision blocks (mask path)") {
            var h = embedded
            for (i, block) in model.visual.blocks.enumerated() {
                let mask = model.visual.fullAttnLayers.contains(i)
                    ? nil : windowMask
                h = block(h, mask: mask)
            }
            return h
        }
        if let plan {
            timeStage("vision blocks (batched windowed)") {
                var h = embedded
                for (i, block) in model.visual.blocks.enumerated() {
                    if model.visual.fullAttnLayers.contains(i) {
                        h = block(h, mask: nil)
                        continue
                    }
                    // Match Qwen25VLVisionTower.runWindowedBlock:
                    // gather by perm, reshape to per-window batch,
                    // attend without mask, reshape back, invert.
                    let flat = h.squeezed(axis: 0)
                    let grouped = flat.take(plan.perm, axis: 0)
                        .reshaped(
                            plan.numWindows, plan.windowSize, flat.dim(1))
                    let attended = block(grouped, mask: nil)
                    let restored = attended
                        .reshaped(
                            plan.numWindows * plan.windowSize,
                            attended.dim(2))
                        .take(plan.invPerm, axis: 0)
                    h = restored.expandedDimensions(axis: 0)
                }
                return h
            }
        } else {
            print("  (batched windowed path skipped: ragged grid)")
        }

        // -- Stage 3: the patch merger --
        // Drive the blocks once more so we have a realized hidden
        // state to feed the merger. Uses the (faster) batched path
        // when available so the merger timing is taken against the
        // shape the production tower produces.
        var blocksOut = embedded
        for (i, block) in model.visual.blocks.enumerated() {
            if model.visual.fullAttnLayers.contains(i) {
                blocksOut = block(blocksOut, mask: nil)
            } else if let plan {
                let flat = blocksOut.squeezed(axis: 0)
                let grouped = flat.take(plan.perm, axis: 0)
                    .reshaped(
                        plan.numWindows, plan.windowSize, flat.dim(1))
                let attended = block(grouped, mask: nil)
                let restored = attended
                    .reshaped(
                        plan.numWindows * plan.windowSize,
                        attended.dim(2))
                    .take(plan.invPerm, axis: 0)
                blocksOut = restored.expandedDimensions(axis: 0)
            } else {
                blocksOut = block(blocksOut, mask: windowMask)
            }
        }
        MLX.eval(blocksOut)
        let blocksSqueezed = blocksOut.squeezed(axis: 0)
        timeStage("merger") {
            model.visual.merger(blocksSqueezed)
        }

        // -- Stage 4: full vision tower (sanity: ~ sum of 1..3) --
        timeStage("vision tower (total)") {
            model.visual(
                patchBatch,
                gridHWFull: (gridHMerged * vision.spatialMergeSize,
                             gridWMerged * vision.spatialMergeSize))
        }

        // -- Stage 5: text prefill (transformer stack only) --
        let visionEmbeds = model.visual(
            patchBatch,
            gridHWFull: (gridHMerged * vision.spatialMergeSize,
                         gridWMerged * vision.spatialMergeSize))
        MLX.eval(visionEmbeds)
        let textModel = model.languageModel.model
        let baseEmbeds = textModel.embedTokens(promptArray)
        let injected = Qwen25VLForConditionalGeneration.injectVisionEmbeds(
            inputEmbeds: baseEmbeds,
            visionEmbeds: visionEmbeds,
            imagePadStart: imagePadStart)
        MLX.eval(injected)
        let coords = Qwen25VLPositions.compute(
            tokenIds: prompt32, imageTokenId: cfg.imageTokenId,
            gridHMerged: gridHMerged, gridWMerged: gridWMerged)
        let posT = MLXArray(coords.t)
        let posH = MLXArray(coords.h)
        let posW = MLXArray(coords.w)
        timeStage("text prefill (\(cfg.numHiddenLayers) layers)") {
            let caches = makeKVCaches(numLayers: cfg.numHiddenLayers)
            return textModel(
                inputEmbeds: injected,
                positionsT: posT, positionsH: posH, positionsW: posW,
                caches: caches)
        }

        // -- Stage 6: the vocab projection (lm_head / tied head) --
        // The 3B checkpoint ties the embeddings, so the projection
        // runs through `embedTokens.asLinear`; an untied checkpoint
        // would route through `lmHead`. Time whichever applies, both
        // over all positions and over the last token only - the
        // delta is the Phase 1 last-token-only win.
        let prefillHidden = textModel(
            inputEmbeds: injected,
            positionsT: posT, positionsH: posH, positionsW: posW,
            caches: makeKVCaches(numLayers: cfg.numHiddenLayers))
        MLX.eval(prefillHidden)
        let lastHidden = prefillHidden[
            0..., (prefillHidden.dim(1) - 1)..., 0...]
        MLX.eval(lastHidden)
        if let lmHead = model.languageModel.lmHead {
            timeStage("lm_head (all positions)") { lmHead(prefillHidden) }
            timeStage("lm_head (last token)") { lmHead(lastHidden) }
        } else {
            let embed = model.languageModel.model.embedTokens
            timeStage("tied head (all positions)") {
                embed.asLinear(prefillHidden)
            }
            timeStage("tied head (last token)") {
                embed.asLinear(lastHidden)
            }
        }

        // -- Stage 7: one decode step (KV-cached single token) --
        let decodeCaches = makeKVCaches(numLayers: cfg.numHiddenLayers)
        let prefillLogits = model(
            promptArray, pixelValues: patchBatch,
            imageGridMerged: (gridHMerged, gridWMerged),
            caches: decodeCaches, mropePositionOffset: nil)
        MLX.eval(prefillLogits)
        let frontier = Int(coords.nextPos)
        let decodeTok = MLXArray([Int32(40)]).reshaped(1, 1)
        timeStage("decode step (1 token)") {
            model(
                decodeTok, pixelValues: nil, imageGridMerged: nil,
                caches: decodeCaches, mropePositionOffset: frontier)
        }

        print("=== end profile ===\n")
    }
}
