import XCTest
@testable import KLMRegistry

/// WS3: `ModelAdapter` is the single declarative source of truth for
/// family-specific server routing and chat-template decisions. These
/// tests pin the routing and template for every family so a new
/// family addition is a deliberate, reviewed change here rather than
/// a silently introduced server branch.
final class ModelAdapterTests: XCTestCase {

    // MARK: - Chat routing

    func testQwen25VLRoutesToVisionBridge() {
        XCTAssertEqual(ModelAdapter(family: .qwen25vl).chatRouting, .visionBridge,
            "Qwen 2.5-VL is bridge-backed and must route to the VLM handler")
    }

    func testMoERoutesToMixtureOfExperts() {
        XCTAssertEqual(ModelAdapter(family: .moe).chatRouting, .mixtureOfExperts,
            "MoE must route to the native-or-bridge MoE path")
    }

    func testDenseFamiliesRouteToDenseEngine() {
        let dense: [ModelFamily] = [
            .llama, .qwen, .mistral, .gemma, .gemma4, .phi, .glm,
            .deepseek, .bert, .reranker,
        ]
        for family in dense {
            XCTAssertEqual(ModelAdapter(family: family).chatRouting, .denseEngine,
                "\(family) has no bridge handler and must fall through to the dense path")
        }
    }

    /// Every family must resolve to a routing - the `switch` in
    /// `chatRouting` is exhaustive, so a newly added `ModelFamily`
    /// case fails to compile until it is given a routing. This test
    /// guards the *intent* (a real value, not a crash) for all cases.
    func testEveryFamilyHasARouting() {
        for family in ModelFamily.allCases {
            let routing = ModelAdapter(family: family).chatRouting
            XCTAssertTrue(ChatRouting.allCases.contains(routing),
                "\(family) resolved to an unexpected routing \(routing)")
        }
    }

    // MARK: - Image-input requirement

    func testOnlyQwen25VLRequiresImageInput() {
        XCTAssertTrue(ModelAdapter(family: .qwen25vl).requiresImageInput,
            "Qwen 2.5-VL has no text-only runtime; a text-only turn must be refused")
        for family in ModelFamily.allCases where family != .qwen25vl {
            XCTAssertFalse(ModelAdapter(family: family).requiresImageInput,
                "\(family) must not force an image on every request")
        }
    }

    /// A family that requires image input must be the one that routes
    /// to the vision bridge - the refusal in `dispatchFamilyChat`
    /// depends on these two facts agreeing.
    func testRequiresImageInputImpliesVisionBridge() {
        for family in ModelFamily.allCases {
            let adapter = ModelAdapter(family: family)
            if adapter.requiresImageInput {
                XCTAssertEqual(adapter.chatRouting, .visionBridge,
                    "\(family) requires an image but does not route to the VLM bridge")
            }
        }
    }

    // MARK: - Chat template policy

    func testNativeToolTemplateFamilies() {
        XCTAssertEqual(ModelAdapter(family: .gemma4).chatTemplate, .gemma4)
        XCTAssertEqual(ModelAdapter(family: .llama).chatTemplate, .llama)
        XCTAssertEqual(ModelAdapter(family: .qwen).chatTemplate, .qwen)
    }

    func testMoEUsesQwenTemplate() {
        XCTAssertEqual(ModelAdapter(family: .moe).chatTemplate, .qwen,
            "the only native MoE runtime (Qwen 3 MoE) uses the Qwen tool template")
    }

    func testFallbackFamiliesUseHermesTemplate() {
        let hermes: [ModelFamily] = [
            .mistral, .gemma, .phi, .glm, .deepseek, .bert,
            .qwen25vl, .reranker,
        ]
        for family in hermes {
            XCTAssertEqual(ModelAdapter(family: family).chatTemplate, .hermes,
                "\(family) has no native tool template and must use the Hermes fallback")
        }
    }

    /// A family that declares the `.tools` capability and has a
    /// native dense path must have a non-Hermes (native) tool
    /// template - otherwise the registry advertises native tool
    /// support it does not back. (`.moe` -> `.qwen` satisfies this;
    /// `.qwen25vl` declares `.tools` but is bridge-backed, so it is
    /// excused.) `ModelAdapter` and `ModelCapabilities` are sibling
    /// tables; this test pins that they stay consistent.
    func testNativeToolCapabilityAgreesWithTemplate() {
        for family in ModelFamily.allCases where family != .qwen25vl {
            if ModelCapabilities.capabilities(for: family).contains(.tools) {
                XCTAssertNotEqual(ModelAdapter(family: family).chatTemplate, .hermes,
                    "\(family) declares the tools capability but falls back to the Hermes template")
            }
        }
    }
}
