import XCTest
@testable import KrillGrammar
import KrillTooling

/// The tool-name grammar makes an unknown tool name unrepresentable at sampling
/// time. These tests pin three things:
///
///  1. it stays out of the way (prose and argument values decode freely),
///  2. it arms after each family's sentinel and rejects any name outside the
///     offered set,
///  3. it cannot be tricked into arming by a `"name"` that appears inside a
///     string value - the failure that would corrupt legitimate output.
final class ToolNameGrammarTests: XCTestCase {

    private let names = ["read_file", "write_file", "bash"]

    private func automaton(_ sentinels: [String]) -> ToolNameAutomaton {
        ToolNameAutomaton(sentinels: sentinels, nameKey: "name", names: names)
    }

    /// Feed `text` from the initial state. Returns nil if the grammar rejected.
    private func run(_ a: ToolNameAutomaton, _ text: String) -> ToolNameAutomaton.State? {
        a.advance(a.initialState, piece: text)
    }

    /// True when the grammar is currently constraining the name slot.
    private func isConstrainingName(_ s: ToolNameAutomaton.State?) -> Bool {
        if case .inName = s { return true }
        return false
    }

    // MARK: - Stays out of the way

    func testProseDecodesUnconstrained() {
        let a = automaton(["<tool_call>"])
        let prose = "Sure. I'll read the file first, then summarise what fizzbuzz returns."
        XCTAssertNotNil(run(a, prose), "ordinary prose must never be rejected")
    }

    func testProseContainingTheWordNameIsFine() {
        let a = automaton(["<tool_call>"])
        XCTAssertNotNil(run(a, #"The variable "name" holds a string."#))
    }

    func testEOSAllowedInProseButNotMidName() {
        let a = automaton(["<tool_call>"])
        XCTAssertTrue(a.isComplete(a.initialState), "the model must be free to answer and stop")

        let mid = run(a, #"<tool_call>{"name": "read"#)
        XCTAssertTrue(isConstrainingName(mid))
        XCTAssertFalse(a.isComplete(mid!), "stopping mid-name would emit a truncated tool name")
    }

    // MARK: - Per family

    /// Every family that carries an unambiguous sentinel must arm on it, and
    /// must reject a name outside the offered set at the FIRST wrong character.
    func testEachFamilyArmsAndRejectsAnUnknownName() {
        // (format, a realistic call opening for that family)
        let cases: [(ToolCalling.ToolFormat, String)] = [
            (.hermes,  #"<tool_call>{"name": ""#),
            (.qwen,    #"<tool_call>{"name": ""#),
            (.gemma4,  #"<tool_call>{"name": ""#),
            (.mistral, #"[TOOL_CALLS][{"name": ""#),
            (.phi,     #"<|tool_call|>[{"name": ""#),
            (.llama,   #"<|python_tag|>{"name": ""#),
        ]
        for (format, opening) in cases {
            let sentinels = ToolCallSentinels.sentinels(for: format)
            XCTAssertFalse(sentinels.isEmpty, "\(format) should carry a sentinel")
            let a = automaton(sentinels)

            let armed = run(a, opening)
            XCTAssertTrue(
                isConstrainingName(armed),
                "\(format): the name slot should be constrained after its sentinel")

            // An offered name is spellable...
            XCTAssertNotNil(a.advance(armed!, piece: #"read_file""#), "\(format): offered name")

            // ...and `Read` is not: capital R cannot start any offered name, so
            // the sampler could never have chosen it.
            XCTAssertNil(a.advance(armed!, piece: "R"),
                         "\(format): a name outside the offered set must be unrepresentable")
        }
    }

    func testGemmaLegacySentinelAlsoArms() {
        let a = automaton(ToolCallSentinels.sentinels(for: .gemma4))
        XCTAssertTrue(isConstrainingName(run(a, #"<|tool_call|>[{"name": ""#)))
    }

    /// `.pythonic` has no marker before the name, so it is deliberately not
    /// constrained. Asserting the policy keeps the exclusion deliberate rather
    /// than something that silently regresses.
    func testFamiliesWithoutASentinelAreNotConstrained() {
        XCTAssertTrue(ToolCallSentinels.sentinels(for: .pythonic).isEmpty)
        XCTAssertFalse(ToolCallSentinels.supportsNameConstraint(.pythonic))
        XCTAssertTrue(ToolCallSentinels.supportsNameConstraint(.hermes))
    }

    // MARK: - Cannot be tricked into arming

    func testNameKeyInsideAStringValueDoesNotArm() {
        // The agent writes a JSON file whose CONTENT contains a "name" key. If
        // the grammar armed here it would constrain the user's data.
        let a = automaton(["<tool_call>"])
        let call = #"<tool_call>{"name": "write_file", "arguments": {"content": "{\"name\": \"Read\"}"#
        let state = run(a, call)
        XCTAssertNotNil(state, "escaped content must not be rejected")
        XCTAssertFalse(
            isConstrainingName(state),
            "a \"name\" inside a string value must not arm the constraint")
    }

    func testStringValueEqualToTheNameKeyRecoversWithoutConstraining() {
        // `"key": "name",` - the VALUE is literally `name`. The grammar may
        // briefly consider it, but the missing colon must drop it back to
        // scanning rather than constraining the next string.
        let a = automaton(["<tool_call>"])
        let state = run(a, #"<tool_call>{"key": "name", "other": "Read"#)
        XCTAssertNotNil(state)
        XCTAssertFalse(isConstrainingName(state))
    }

    func testConstraintDisarmsAfterTheNameSoArgumentsAreFree() {
        let a = automaton(["<tool_call>"])
        // Arguments may contain anything, including words that are not tools.
        let full = #"<tool_call>{"name": "bash", "arguments": {"command": "Read the docs"}}"#
        XCTAssertNotNil(run(a, full), "argument values must decode unconstrained")
    }

    func testSecondCallInTheSameTurnIsAlsoConstrained() {
        let a = automaton(["<tool_call>"])
        let first = #"<tool_call>{"name": "bash", "arguments": {}}</tool_call>"#
        let state = run(a, first + #"<tool_call>{"name": ""#)
        XCTAssertTrue(isConstrainingName(state), "each call's name must be constrained")
        XCTAssertNil(a.advance(state!, piece: "R"))
    }

    /// Performance invariant, not a behaviour one. Every distinct grammar state
    /// costs a full vocabulary scan to build its mask, so the scanner must not
    /// accumulate arbitrary text: long argument values have to collapse onto a
    /// single state. Without this the mask rebuilds once per character of
    /// everything the model writes.
    func testStateSpaceStaysBoundedAcrossLongArgumentValues() {
        let a = automaton(["<tool_call>"])
        let base = #"<tool_call>{"name": "bash", "arguments": {"command": ""#
        let short = run(a, base + String(repeating: "x", count: 50))
        let long = run(a, base + String(repeating: "y", count: 500))
        XCTAssertNotNil(short)
        XCTAssertEqual(
            short, long,
            "argument text of any length must collapse to one state")
    }

    func testSentinelSplitAcrossTokensStillArms() {
        // Token boundaries are arbitrary; the sentinel often spans several.
        let a = automaton(["<tool_call>"])
        var s = a.initialState
        for piece in ["<tool", "_call", ">", #"{"name": ""#] {
            guard let next = a.advance(s, piece: piece) else {
                return XCTFail("rejected at piece \(piece)")
            }
            s = next
        }
        XCTAssertTrue(isConstrainingName(s))
    }
}
