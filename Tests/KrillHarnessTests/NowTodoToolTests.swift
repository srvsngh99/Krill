import Foundation
import XCTest
@testable import KrillHarness

final class NowTodoToolTests: XCTestCase {
    func testNowToolReportsCurrentInstant() async {
        let before = Int(Date().timeIntervalSince1970)
        let result = await NowTool().run(argumentsJSON: "{}")
        let after = Int(Date().timeIntervalSince1970)

        XCTAssertFalse(result.isError)
        let unixLine = result.content.split(separator: "\n").first { $0.hasPrefix("unix: ") }
        let unix = unixLine.flatMap { Int($0.dropFirst("unix: ".count)) }
        XCTAssertNotNil(unix)
        XCTAssertTrue((before ... after).contains(unix ?? -1))
        XCTAssertTrue(result.content.contains("iso8601: "))
    }

    func testTodoToolReplacesAndRendersChecklist() async {
        let todo = TodoTool()

        let empty = await todo.run(argumentsJSON: "{}")
        XCTAssertEqual(empty.content, "(todo list is empty)")

        let set = await todo.run(argumentsJSON: """
        {"items":[{"text":"survey repo"},{"text":"write fix","done":false}]}
        """)
        XCTAssertTrue(set.content.hasPrefix("Todo (0/2 done):"))
        XCTAssertTrue(set.content.contains("[ ] survey repo"))

        let progress = await todo.run(argumentsJSON: """
        {"items":[{"text":"survey repo","done":true},{"text":"write fix"}]}
        """)
        XCTAssertTrue(progress.content.hasPrefix("Todo (1/2 done):"))
        XCTAssertTrue(progress.content.contains("[x] survey repo"))
        XCTAssertTrue(progress.content.contains("[ ] write fix"))

        // No items → view without mutating.
        let view = await todo.run(argumentsJSON: "{}")
        XCTAssertEqual(view.content, progress.content)

        // Blank rows are dropped rather than rendered as empty checklist lines.
        let cleaned = await todo.run(argumentsJSON: """
        {"items":[{"text":"  "},{"text":"only real step"}]}
        """)
        XCTAssertTrue(cleaned.content.hasPrefix("Todo (0/1 done):"))
    }

    func testEnvironmentContextLineCarriesAmbientFacts() {
        let line = AgentEnvironment.contextLine(modelName: "test-model")
        XCTAssertTrue(line.hasPrefix("Environment: "))
        XCTAssertTrue(line.contains("cwd "))
        XCTAssertTrue(line.contains("model test-model"))
        let year = Calendar.current.component(.year, from: Date())
        XCTAssertTrue(line.contains(String(year)))
    }
}
