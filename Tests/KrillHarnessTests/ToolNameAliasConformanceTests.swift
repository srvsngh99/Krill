import XCTest
@testable import KrillHarness
@testable import KrillTooling

/// The alias table in `ToolCalling` maps another harness's tool vocabulary
/// (Claude Code's `Read` / `Edit` / `Bash`) onto this one's names. Its keys are
/// external and may be anything, but every VALUE has to be a tool this harness
/// actually registers - an alias pointing at a name no tool answers to is dead
/// weight that silently never fires, and the symptom (an agent dying on
/// "unknown tool") looks identical to having no alias at all.
///
/// Renaming a tool is the realistic way that happens, so pin the table against
/// the real registry: rename `read_file` and this test fails instead of the
/// alias quietly rotting.
final class ToolNameAliasConformanceTests: XCTestCase {

    /// Every tool type the CLI can register, across `krill code` and the chat
    /// TUI's agent mode. Kept as one union deliberately: an alias may legitimately
    /// target a tool that some sessions do not offer (resolution is gated on the
    /// offered set at call time), but it must always name a real tool.
    private func allRegisteredToolNames() -> Set<String> {
        let tools: [any Tool] = [
            ReadTool(), ListTool(), GlobTool(), GrepTool(),
            WebFetchTool(), WebSearchTool(),
            EditTool(), MultiEditTool(), WriteTool(),
            BashTool(), DispatchTool(queue: SpawnQueue()),
        ]
        return Set(ToolRegistry(tools).names)
    }

    func testEveryAliasTargetIsARegisteredTool() {
        let registered = allRegisteredToolNames()
        for (alias, target) in ToolCalling.agentToolAliases {
            XCTAssertTrue(
                registered.contains(target),
                """
                Alias "\(alias)" -> "\(target)" does not name a registered tool.
                Registered: \(registered.sorted().joined(separator: ", ")).
                Fix the alias target, or drop the entry if the tool is gone.
                """)
        }
    }

    /// An alias must never shadow a real tool name: if a tool is genuinely called
    /// `read`, the exact-match rule has to win and the alias must not redirect it.
    func testNoAliasKeyShadowsARegisteredToolName() {
        let registered = allRegisteredToolNames()
        for alias in ToolCalling.agentToolAliases.keys {
            XCTAssertFalse(
                registered.contains(alias),
                "Alias key \"\(alias)\" collides with a registered tool of the same name.")
        }
    }

    /// The offered set is the only authority: aliases resolve within it and never
    /// outside it. Guards the gate that stops a mismatch becoming a wrong action.
    func testAliasResolvesOnlyWithinTheOfferedSet() {
        let offered = ["read_file", "bash"]
        let calls = [ToolCalling.ParsedToolCall(name: "Read", argumentsJSON: "{}")]
        XCTAssertEqual(
            ToolCalling.canonicalizeNames(calls, known: offered)[0].name, "read_file")

        // `write_file` is a real tool but is NOT offered here, so `Write` must not
        // resolve: dispatching an unoffered tool would be worse than failing.
        let writeCalls = [ToolCalling.ParsedToolCall(name: "Write", argumentsJSON: "{}")]
        XCTAssertEqual(
            ToolCalling.canonicalizeNames(writeCalls, known: offered)[0].name, "Write")
    }
}
