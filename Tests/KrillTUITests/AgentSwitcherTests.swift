import XCTest
@testable import KrillTUI

final class AgentSwitcherTests: XCTestCase {

    private func sample() -> AgentSwitcher {
        AgentSwitcher(entries: [
            .init(id: nil, title: "main", status: "agent:plan"),
            .init(id: 1, title: "[1] explore", status: "running 3s"),
            .init(id: 2, title: "[2] tests", status: "done"),
        ])
    }

    func testStartsOnCurrentSession() {
        let sw = AgentSwitcher(entries: sample().entries, current: 2)
        XCTAssertEqual(sw.current?.id, 2)
    }

    func testDefaultsToMainRow() {
        XCTAssertEqual(sample().current?.id, nil)
        XCTAssertEqual(sample().current?.title, "main")
    }

    func testCycleWraps() {
        var sw = sample()
        sw.selectPrevious()                 // wrap to last
        XCTAssertEqual(sw.current?.id, 2)
        sw.selectNext()                     // wrap back to first
        XCTAssertEqual(sw.current?.id, nil)
        sw.selectNext()
        XCTAssertEqual(sw.current?.id, 1)
    }

    func testEmptyIsSafe() {
        var sw = AgentSwitcher(entries: [])
        XCTAssertTrue(sw.isEmpty)
        XCTAssertNil(sw.current)
        sw.selectNext(); sw.selectPrevious()   // no crash
    }
}
