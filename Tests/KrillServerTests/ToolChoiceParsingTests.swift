import XCTest
@testable import KrillServer
import KrillTooling

/// `ServerParsing.parseToolChoice` lives in KrillServer (it parses the wire
/// request), while the `ServerToolChoice` type it returns now lives in
/// KrillTooling. This case stayed behind when `ForcedToolCallTests` moved to
/// KrillToolingTests because it exercises that KrillServer-only parser.
final class ToolChoiceParsingTests: XCTestCase {

    func testToolChoiceParsing() {
        XCTAssertEqual(ServerParsing.parseToolChoice("required"), .required)
        XCTAssertEqual(ServerParsing.parseToolChoice("none"), .none)
        XCTAssertEqual(ServerParsing.parseToolChoice("auto"), .auto)
        XCTAssertEqual(ServerParsing.parseToolChoice(nil), .auto)
        XCTAssertEqual(
            ServerParsing.parseToolChoice(["type": "function", "function": ["name": "add"]]),
            .function("add"))
    }
}
