import XCTest
@testable import KrillCore

/// Pins the `loadModel` architecture-detection table
/// (`detectedArchitectureID`). These run without a checkpoint -- detection is
/// a pure function of the config's `architectures` / `model_type` strings --
/// so they are the regression net for the table's load-bearing ordering:
/// "a generic rule must not shadow a more specific one" (e.g. the catch-all
/// `qwen` rule must not steal a `qwen3_moe` config).
final class ArchitectureDetectionTests: XCTestCase {

    private func id(arch: String = "", modelType: String = "") -> String {
        detectedArchitectureID(
            architectures: arch.isEmpty ? [] : [arch], modelType: modelType)
    }

    // MARK: - Each family resolves, via arch string and via model_type

    func testQwen25VLDetection() {
        XCTAssertEqual(id(arch: "Qwen2_5_VLForConditionalGeneration"), "qwen2_5_vl")
        XCTAssertEqual(id(arch: "Qwen2VLForConditionalGeneration"), "qwen2_5_vl")
        XCTAssertEqual(id(modelType: "qwen2_5_vl"), "qwen2_5_vl")
        XCTAssertEqual(id(modelType: "qwen2_vl"), "qwen2_5_vl")
    }

    func testLlavaDetection() {
        // llava's text backbone is Llama, but its arch string is
        // `LlavaForConditionalGeneration` (no "llama" substring), so the rule
        // must claim it explicitly and BEFORE the generic llama rule.
        XCTAssertEqual(id(arch: "LlavaForConditionalGeneration"), "llava")
        XCTAssertEqual(id(modelType: "llava"), "llava")
    }

    func testMllamaDetection() {
        // Llama-3.2-Vision. Its text backbone is Llama, but the arch is
        // `MllamaForConditionalGeneration` and model_type "mllama"; it must be
        // claimed before the generic llama rule and route to loadLlamaVision.
        XCTAssertEqual(id(arch: "MllamaForConditionalGeneration"), "mllama")
        XCTAssertEqual(id(modelType: "mllama"), "mllama")
    }

    func testRerankerRejectionRule() {
        XCTAssertEqual(id(arch: "BertForSequenceClassification"), "reranker")
        XCTAssertEqual(id(arch: "XLMRobertaCrossEncoder"), "reranker")
    }

    func testGemma4Detection() {
        XCTAssertEqual(id(arch: "Gemma4ForCausalLM"), "gemma4")
        XCTAssertEqual(id(modelType: "gemma4_text"), "gemma4")
        XCTAssertEqual(id(modelType: "gemma4"), "gemma4")
    }

    func testGLMDetection() {
        XCTAssertEqual(id(arch: "ChatGLMModel"), "glm")
        XCTAssertEqual(id(arch: "GlmForCausalLM"), "glm")
        XCTAssertEqual(id(modelType: "chatglm"), "glm")
    }

    func testGLM4Detection() {
        // GLM-4-0414 / GLM-Z1 must route to the new glm4 rule, NOT the legacy
        // glm rule (whose `arch.contains("glm")` matcher would otherwise claim
        // it first). The glm4 rule is ordered before glm for exactly this.
        XCTAssertEqual(id(arch: "Glm4ForCausalLM"), "glm4")
        XCTAssertEqual(id(modelType: "glm4"), "glm4")
        // Glm4MoeForCausalLM (GLM-4.5 / GLM-MoE) also contains "glm4" but is a
        // sparse-MoE arch the dense glm4 runtime cannot serve. It must NOT route
        // to glm4; the `moe` guard sends it on to the legacy glm rule instead.
        XCTAssertNotEqual(id(arch: "Glm4MoeForCausalLM"), "glm4")
        XCTAssertEqual(id(arch: "Glm4MoeForCausalLM"), "glm")
    }

    func testMoEFamilies() {
        XCTAssertEqual(id(arch: "Qwen3MoeForCausalLM"), "qwen3_moe")
        XCTAssertEqual(id(modelType: "qwen3_moe"), "qwen3_moe")
        XCTAssertEqual(id(arch: "MixtralForCausalLM"), "mixtral")
        XCTAssertEqual(id(arch: "Qwen2MoeForCausalLM"), "qwen2_moe")
        XCTAssertEqual(id(modelType: "qwen2_moe"), "qwen2_moe")
        XCTAssertEqual(id(arch: "OlmoeForCausalLM"), "olmoe")
    }

    func testDeepSeekDetection() {
        XCTAssertEqual(id(arch: "DeepseekV2ForCausalLM"), "deepseek")
        XCTAssertEqual(id(modelType: "deepseek_v2"), "deepseek")
        // V3 still routes to the deepseek rule (loadDeepSeek then rejects the
        // absorbed-MLA layout); it must NOT fall through to the fallback.
        XCTAssertEqual(id(modelType: "deepseek_v3"), "deepseek")
    }

    func testDenseFamilies() {
        XCTAssertEqual(id(arch: "LlamaForCausalLM"), "llama")
        XCTAssertEqual(id(arch: "Qwen2ForCausalLM"), "qwen")
        XCTAssertEqual(id(modelType: "qwen2"), "qwen")
        XCTAssertEqual(id(arch: "MistralForCausalLM"), "mistral")
        XCTAssertEqual(id(arch: "GemmaForCausalLM"), "gemma")
        XCTAssertEqual(id(arch: "Phi3ForCausalLM"), "phi")
        XCTAssertEqual(id(modelType: "phi3"), "phi")
    }

    func testQwen35Detection() {
        // Ornith-9B. Its arch `Qwen3_5ForConditionalGeneration` and model_type
        // `qwen3_5` (or the nested `qwen3_5_text`) must route to the hybrid
        // qwen3_5 loader, NOT the generic dense qwen rule.
        XCTAssertEqual(id(arch: "Qwen3_5ForConditionalGeneration"), "qwen3_5")
        XCTAssertEqual(id(modelType: "qwen3_5"), "qwen3_5")
        XCTAssertEqual(id(modelType: "qwen3_5_text"), "qwen3_5")
    }

    func testSpecializedRejectionRule() {
        XCTAssertEqual(id(arch: "WhisperForConditionalGeneration"), "specialized")
        XCTAssertEqual(id(modelType: "whisper"), "specialized")
    }

    func testUnknownArchitectureFallsBackToLlama() {
        XCTAssertEqual(id(arch: "SomeBrandNewForCausalLM"), "fallback")
        XCTAssertEqual(id(), "fallback")  // empty config
    }

    // MARK: - Order-sensitivity: a specific rule wins over a later generic one

    /// A Qwen3-MoE arch string also contains the substring "qwen", and a
    /// Gemma 4 arch also contains "gemma" -- if the generic rules ran first
    /// these would mis-load as dense Qwen / Gemma. Lock specific-before-generic.
    func testSpecificRulesWinOverGenericSubstringRules() {
        // "qwen3moeforcausallm" contains both "qwen3moe" and "qwen".
        XCTAssertEqual(id(arch: "Qwen3MoeForCausalLM"), "qwen3_moe")
        // "qwen2moeforcausallm" contains both "qwen2moe" and "qwen".
        XCTAssertEqual(id(arch: "Qwen2MoeForCausalLM"), "qwen2_moe")
        // "gemma4forcausallm" contains both "gemma4" and "gemma".
        XCTAssertEqual(id(arch: "Gemma4ForCausalLM"), "gemma4")
        // "qwen3_5forconditionalgeneration" contains "qwen"; the hybrid rule
        // (placed before generic qwen) must claim it, not the dense loader.
        XCTAssertEqual(id(arch: "Qwen3_5ForConditionalGeneration"), "qwen3_5")
        // model_type only: "qwen3_moe" must not be stolen by the qwen prefix rule.
        XCTAssertEqual(id(modelType: "qwen3_moe"), "qwen3_moe")
    }

    /// Every rule id in the table is reachable except via its own matcher,
    /// and the table ends with the catch-all `fallback`.
    func testTableIsOrderedAndFallbackIsLast() {
        XCTAssertEqual(architectureRules.last?.id, "fallback")
        XCTAssertTrue(architectureRules.last?.matches("anything", "anything") ?? false,
            "the last rule must match any input")
        // ids are unique.
        let ids = architectureRules.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "rule ids must be unique")
    }
}
