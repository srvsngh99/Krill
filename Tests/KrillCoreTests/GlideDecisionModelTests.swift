import XCTest
import KrillRuntime
@testable import KrillCore

final class GlideDecisionModelTests: XCTestCase {
    func testPromptMatchesReference() {
        let prompt = GlideDecisionModel.prompt(
            state: "{}", instructions: "Pick the intent.",
            options: ["opt_0: billing", "opt_1  "], codes: ["A", "B", "C"])
        XCTAssertEqual(prompt, """
        <|im_start|>user
        Context:
        {}

        Question: Pick the intent.

        Options:
        (A) opt_0: billing
        (B) opt_1

        Think it through briefly, then answer with the option letter only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        Answer: (
        """)
    }

    func testWrapKeepsHeadAndAnswerCue() {
        XCTAssertEqual(GlideDecisionModel.wrap([3, 4, 5], cls: 1, sep: 2, limit: 5, head: 1), [1, 3, 4, 5, 2])
        XCTAssertEqual(GlideDecisionModel.wrap([3, 4, 5, 6, 7], cls: 1, sep: 2, limit: 5, head: 1), [1, 3, 6, 7, 2])
    }

    func testReferenceParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["KRILL_GLIDE_MODEL_PATH"], let dataPath = env["KRILL_GLIDE_PARITY_ROWS"] else {
            throw XCTSkip("Run scripts/glide_parity.py with the model, inputs and results paths")
        }
        try MLXMetalRuntime.validateForNativeInference()
        let directory = URL(fileURLWithPath: path)
        XCTAssertTrue(DecisionModel.isBundle(directory))
        let model = try await DecisionModel.load(directory: directory)
        XCTAssertTrue(model.isGlide)
        try checkDecisionReferenceParity(model, name: "GLiDE", dataPath: dataPath)
    }
}
