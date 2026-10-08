import XCTest
@testable import KrillRegistry

final class ModelCapabilitiesTests: XCTestCase {

    // MARK: - Text generation

    func testDenseTextFamiliesAllDeclareTextGeneration() {
        for family: ModelFamily in [.llama, .qwen, .mistral, .gemma, .phi, .glm, .deepseek] {
            XCTAssertTrue(
                ModelCapabilities.capabilities(for: family).contains(.textGeneration),
                "\(family) is a causal text LM and must declare textGeneration")
        }
        XCTAssertTrue(ModelCapabilities.capabilities(for: .gemma4).contains(.textGeneration))
    }

    func testBertDoesNotDeclareTextGeneration() {
        let caps = ModelCapabilities.capabilities(for: .bert)
        XCTAssertFalse(caps.contains(.textGeneration),
            "BERT-class encoders are not causal LMs; server must reject /api/generate on them")
        XCTAssertTrue(caps.contains(.embeddings))
    }

    // MARK: - Multimodal capabilities

    func testOnlyGemma4DeclaresVisionAndAudio() {
        for family: ModelFamily in [.llama, .qwen, .mistral, .gemma, .phi, .glm, .deepseek, .bert] {
            let caps = ModelCapabilities.capabilities(for: family)
            XCTAssertFalse(caps.contains(.visionInput),
                "\(family) must not declare visionInput")
            XCTAssertFalse(caps.contains(.audioInput),
                "\(family) must not declare audioInput")
        }
        let gemma4Caps = ModelCapabilities.capabilities(for: .gemma4)
        XCTAssertTrue(gemma4Caps.contains(.visionInput))
        XCTAssertTrue(gemma4Caps.contains(.audioInput))
    }

    // MARK: - Support tier

    func testEveryFamilyHasAnExplicitSupportTier() {
        // The CaseIterable + exhaustive switch contract: any new
        // ModelFamily case must add its own arm to supportTier(for:),
        // otherwise this test (and the compiler) flags it.
        for family in ModelFamily.allCases {
            let tier = ModelCapabilities.supportTier(for: family)
            XCTAssertNotNil(SupportTier(rawValue: tier.rawValue),
                "\(family) returned a malformed support tier")
        }
    }

    // MARK: - Native tool template

    func testNativeToolTemplateMatchesShippedFamilies() {
        // The set of families with native tool chat templates. Originally
        // landed in PR #23 (commit 825d1b3); mistral + phi were promoted to
        // tool-capable in the tool-calling standardization (decision 0001),
        // since they ship native inject templates + parsers
        // (injectMistral/extractMistral, injectPhi/extractPhi). This test pins
        // the set so an unintended regression is caught.
        XCTAssertTrue(ModelCapabilities.hasNativeToolTemplate(.gemma4))
        XCTAssertTrue(ModelCapabilities.hasNativeToolTemplate(.llama))
        XCTAssertTrue(ModelCapabilities.hasNativeToolTemplate(.qwen))
        XCTAssertTrue(ModelCapabilities.hasNativeToolTemplate(.mistral))
        XCTAssertTrue(ModelCapabilities.hasNativeToolTemplate(.phi))
        XCTAssertFalse(ModelCapabilities.hasNativeToolTemplate(.glm))
        XCTAssertFalse(ModelCapabilities.hasNativeToolTemplate(.bert))
    }

    func testHasNativeToolTemplateMatchesToolsCapability() {
        // The helper is defined in terms of the capability set; this
        // test catches future drift where one is updated without the
        // other (e.g. a family gets a parity-tested tool template but
        // hasNativeToolTemplate falls out of sync).
        for family in ModelFamily.allCases {
            XCTAssertEqual(
                ModelCapabilities.hasNativeToolTemplate(family),
                ModelCapabilities.capabilities(for: family).contains(.tools),
                "hasNativeToolTemplate(\(family)) must match .tools in the capability set")
        }
    }

    // MARK: - EmbeddingGemma 2

    func testEmbeddingGemma2RoutesToEmbeddingFamily() {
        let cfg: [String: Any] = [
            "architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .bert,
                       "must not route to the causal .gemma/.gemma4 families")
        XCTAssertEqual(ModelFamily.detect(from: ["model_type": "embedding_gemma2"]), .bert)
        XCTAssertEqual(ModelCapabilities.capabilities(for: .bert), [.embeddings])
    }

    func testEmbeddingGemma2AliasResolves() {
        let hit = AliasMap.allAliases.first { $0.shortName == "embeddinggemma-2" }
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.model.repo, "google/embeddinggemma-2")
        XCTAssertEqual(hit?.model.family, .bert)
        XCTAssertEqual(hit?.model.context, 8192)
    }

    func testEmbeddingGemma2QuantizedAliasesResolve() {
        for (name, quant) in [("embeddinggemma-2-mxfp8", "mxfp8"), ("embeddinggemma-2-nvfp4", "nvfp4")] {
            let hit = AliasMap.allAliases.first { $0.shortName == name }
            XCTAssertNotNil(hit, name)
            XCTAssertEqual(hit?.model.repo, "srv-sngh/embeddinggemma-2-mlx-" + quant)
            XCTAssertEqual(hit?.model.quant, quant)
            XCTAssertEqual(hit?.model.family, .bert)
            XCTAssertEqual(hit?.model.context, 8192)
        }
    }

    // MARK: - Ollama tag mapping

    func testOllamaTagMappingIsStableAndDistinct() {
        let tags = Capability.allCases.map(\.ollamaTag)
        XCTAssertEqual(Set(tags).count, tags.count,
            "ollamaTag values must be unique across capabilities")
        XCTAssertEqual(Capability.textGeneration.ollamaTag, "completion",
            "textGeneration must emit Ollama's `completion` alias for drop-in compatibility")
        XCTAssertEqual(Capability.visionInput.ollamaTag, "vision")
        XCTAssertEqual(Capability.audioInput.ollamaTag, "audio")
        XCTAssertEqual(Capability.embeddings.ollamaTag, "embedding")
    }
}
