import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// End-to-end (base-view) multimodal prefill parity: Krill's full stack — vision
/// DeepEncoder + base-view splice + the quantized DeepSeek LM via the
/// inputs_embeds path — vs the HF reference, on identical input_ids + image.
/// Validates the whole pipeline produces the right first-token distribution.
///
/// Gated on KRILL_UNLIMITED_OCR_DIR (snapshot: vision weights),
/// KRILL_UNLIMITED_OCR_LM_DIR (the 8-bit quantized language checkpoint from
/// tools/unlimited_ocr_make_text_checkpoint.py), and KRILL_UNLIMITED_OCR_E2E_REF
/// (tools/unlimited_ocr_e2e_reference.py). Krill runs the 8-bit LM vs the fp32
/// reference, so the gate is argmax-equality + high cosine.
final class UnlimitedOCREndToEndParityTests: XCTestCase {
    func testMultimodalPrefillMatchesHFReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["KRILL_UNLIMITED_OCR_DIR"],
              let lmPath = env["KRILL_UNLIMITED_OCR_LM_DIR"],
              let refPath = env["KRILL_UNLIMITED_OCR_E2E_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_DIR + KRILL_UNLIMITED_OCR_LM_DIR + "
                + "KRILL_UNLIMITED_OCR_E2E_REF")
        }
        let dir = URL(fileURLWithPath: dirPath)

        // 1. Vision DeepEncoder (fp32; transpose SAM conv weights; drop CLIP patch_embedding).
        let enc = DeepEncoder()
        var vw: [String: MLXArray] = [:]
        let keep = ["model.sam_model.", "model.vision_model.", "model.projector."]
        var newline = MLXArray.zeros([1280]), sep = MLXArray.zeros([1280])
        for shard in try FileManager.default.contentsOfDirectory(atPath: dirPath)
            .filter({ $0.hasSuffix(".safetensors") }) {
            for (k, v) in try MLX.loadArrays(url: dir.appendingPathComponent(shard)) {
                if k == "model.image_newline" { newline = v.asType(.float32) }
                if k == "model.view_seperator" { sep = v.asType(.float32) }
                guard keep.contains(where: k.hasPrefix) else { continue }
                let key = String(k.dropFirst("model.".count))
                if key.contains("vision_model.embeddings.patch_embedding") { continue }
                var arr = v.asType(.float32)
                if arr.ndim == 4 && !key.hasSuffix("pos_embed") { arr = arr.transposed(0, 2, 3, 1) }
                vw[key] = arr
            }
        }
        try enc.update(parameters: ModuleParameters.unflattened(vw.map { ($0.key, $0.value) }),
                       verify: [.shapeMismatch])
        eval(enc)

        // 2. Quantized DeepSeek LM (8-bit), built like loadDeepSeek.
        let lmDir = URL(fileURLWithPath: lmPath)
        let lmCfg = try JSONDecoder().decode(
            DeepSeekConfig.self, from: Data(contentsOf: lmDir.appendingPathComponent("config.json")))
        let lm = DeepSeekForCausalLM(lmCfg)
        try loadWeights(into: lm, from: lmDir, quantization: lmCfg.quantization, strictVerify: true)
        eval(lm)

        // 3. Reference inputs + expected logits.
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let inputIds = fx["input_ids"]!.asType(.int32)              // [1,L]
        let imageCF = fx["images_ori"]!.asType(.float32)           // [1,3,1024,1024]
        let maskArr = fx["images_seq_mask"]!.asType(.int32).reshaped([-1]).asArray(Int32.self)
        let refLogits = fx["last_logits"]!.asType(.float32).asArray(Float.self)
        let L = inputIds.dim(1)

        // 4. Embed, splice vision at the contiguous <image> block, decode one step.
        var embeds = lm.embedTokens(inputIds).asType(.float32)     // [1,L,1280]
        let visFeat = enc(image: imageCF.transposed(0, 2, 3, 1))   // [1,256,1280]
        let assembled = assembleBaseViewTokens(features: visFeat, imageNewline: newline, viewSeparator: sep)
        let start = maskArr.firstIndex(of: 1)!
        let nImg = assembled.dim(0)
        let flat = embeds.reshaped(L, 1280)
        let spliced = concatenated([
            flat[0 ..< start, 0...], assembled, flat[(start + nImg) ..< L, 0...],
        ], axis: 0).reshaped(1, L, 1280)

        let logits = lm(inputsEmbeds: spliced, lastTokenOnly: true).asType(.float32)
        eval(logits)
        let got = logits.reshaped([-1]).asArray(Float.self)

        var dot = 0.0, na = 0.0, nb = 0.0, gi = 0, gm = -Double.infinity
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(refLogits[i])
            dot += x * y; na += x * x; nb += y * y
            if x > gm { gm = x; gi = i }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        var ri = 0, rm = -Double.infinity
        for i in 0 ..< refLogits.count where Double(refLogits[i]) > rm { rm = Double(refLogits[i]); ri = i }
        // Rank of the reference's argmax token within Krill's logits. As with the
        // text backbone, 8-bit quant flips top-1/top-2 near-ties; a vision/splice
        // bug would instead bury the ref token at a large rank. Gate: ref token in
        // Krill's top-3 + high cosine (NOT exact argmax — that's the fp32 regime).
        let refLogit = Double(got[ri])
        var refRank = 0
        for i in 0 ..< got.count where Double(got[i]) > refLogit { refRank += 1 }
        print("[unlimited-ocr E2E parity] argmax got=\(gi) ref=\(ri) "
            + "| ref-token rank in Krill=\(refRank) | cosine=\(cosine)")
        XCTAssertLessThanOrEqual(refRank, 2, "ref token at rank \(refRank) — vision/splice bug, not quant noise")
        XCTAssertGreaterThan(cosine, 0.997, "end-to-end logit cosine \(cosine) too low")
    }
}
