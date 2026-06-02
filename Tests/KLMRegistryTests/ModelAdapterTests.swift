import XCTest
@testable import KLMRegistry

/// WS3: `ModelAdapter` is the single declarative source of truth for
/// family-specific server routing and chat-template decisions. These
/// tests pin the routing and template for every family so a new
/// family addition is a deliberate, reviewed change here rather than
/// a silently introduced server branch.
final class ModelAdapterTests: XCTestCase {

    // MARK: - Chat routing

    func testMoERoutesToDenseEngine() {
        // Every MoE family is native now and loads through `loadModel` on the
        // dense engine; the mlx-lm sidecar bridge (and the `.mixtureOfExperts`
        // routing case) were removed.
        XCTAssertEqual(ModelAdapter(family: .moe).chatRouting, .denseEngine,
            "MoE must route to the native dense engine path")
    }

    func testDenseFamiliesRouteToDenseEngine() {
        // Qwen 2.5-VL is included: WS5 retired its Python bridge,
        // so it routes through the native dense engine path. `.moe` joined
        // this group once the MoE sidecar was deleted.
        let dense: [ModelFamily] = [
            .llama, .qwen, .qwen25vl, .llava, .mistral, .gemma, .gemma4, .phi,
            .glm, .deepseek, .bert, .reranker, .moe,
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

    /// No family forces an image today: WS5 retired the Qwen 2.5-VL
    /// Python sidecar, and the native VL runtime serves a text-only
    /// turn directly (it just skips the vision tower).
    func testNoFamilyRequiresImageInput() {
        for family in ModelFamily.allCases {
            XCTAssertFalse(ModelAdapter(family: family).requiresImageInput,
                "\(family) must not force an image on every request")
        }
    }

    // MARK: - Chat template policy

    func testNativeToolTemplateFamilies() {
        XCTAssertEqual(ModelAdapter(family: .gemma4).chatTemplate, .gemma4)
        XCTAssertEqual(ModelAdapter(family: .llama).chatTemplate, .llama)
        XCTAssertEqual(ModelAdapter(family: .qwen).chatTemplate, .qwen)
        XCTAssertEqual(ModelAdapter(family: .mistral).chatTemplate, .mistral)
        XCTAssertEqual(ModelAdapter(family: .phi).chatTemplate, .phi)
    }

    func testMoEUsesQwenTemplate() {
        XCTAssertEqual(ModelAdapter(family: .moe).chatTemplate, .qwen,
            "the only native MoE runtime (Qwen 3 MoE) uses the Qwen tool template")
    }

    func testFallbackFamiliesUseHermesTemplate() {
        // Families without a native tool template still fall back to the
        // generic Hermes prompt. (Mistral and Phi gained native adapters
        // 2026-06-01 and are asserted in testNativeToolTemplateFamilies.)
        let hermes: [ModelFamily] = [
            .gemma, .glm, .deepseek, .bert, .qwen25vl, .llava, .reranker,
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
    /// `.qwen25vl` declares `.tools` and resolves to `.hermes`,
    /// which IS Qwen's native `<tool_call>` format, so it is
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
