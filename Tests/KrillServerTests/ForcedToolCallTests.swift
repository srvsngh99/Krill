import XCTest
import KrillGrammar
@testable import KrillServer

/// Front A: grammar-constrained (forced) tool calls. Verifies the schema
/// builder, the bare-JSON parser, and that the built schema actually compiles
/// in the schema grammar (so it can constrain decoding).
final class ForcedToolCallTests: XCTestCase {

    private let weather = ServerToolSpec(
        name: "get_weather",
        description: "Get weather",
        parametersJSON: """
        {"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}
        """)
    private let add = ServerToolSpec(
        name: "add", description: "Add two numbers",
        parametersJSON: #"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}"#)

    private func parse(_ s: String) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: s.data(using: .utf8)!)) as? [String: Any] ?? [:]
    }

    func testSpecificFunctionSchemaPinsNameAndArgs() {
        let schema = ToolCalling.forcedToolCallSchema(
            tools: [weather, add], choice: .function("get_weather"))
        XCTAssertNotNil(schema)
        let obj = parse(schema!)
        let props = obj["properties"] as? [String: Any]
        let nameSchema = props?["name"] as? [String: Any]
        XCTAssertEqual(nameSchema?["const"] as? String, "get_weather")
        // arguments = the tool's own parameter schema (city/days, city required).
        let argsSchema = props?["arguments"] as? [String: Any]
        let argProps = argsSchema?["properties"] as? [String: Any]
        XCTAssertNotNil(argProps?["city"])
        XCTAssertEqual(argsSchema?["required"] as? [String], ["city"])
        XCTAssertEqual(obj["required"] as? [String], ["name", "arguments"])
    }

    func testRequiredMultiToolSchemaEnumeratesNames() {
        let schema = ToolCalling.forcedToolCallSchema(tools: [weather, add], choice: .required)
        let obj = parse(schema!)
        let props = obj["properties"] as? [String: Any]
        let nameSchema = props?["name"] as? [String: Any]
        let names = (nameSchema?["enum"] as? [String]) ?? []
        XCTAssertEqual(Set(names), ["get_weather", "add"])
    }

    func testUnknownForcedFunctionReturnsNil() {
        XCTAssertNil(ToolCalling.forcedToolCallSchema(tools: [weather], choice: .function("nope")))
        XCTAssertNil(ToolCalling.forcedToolCallSchema(tools: [], choice: .required))
        XCTAssertNil(ToolCalling.forcedToolCallSchema(tools: [weather], choice: .auto))
    }

    func testBuiltSchemaCompilesInGrammar() {
        // The whole point: the schema must drive the constrained decoder.
        let schema = ToolCalling.forcedToolCallSchema(tools: [weather], choice: .required)!
        XCTAssertNotNil(SchemaGrammar.compile(schema),
            "forced tool-call schema must compile in the schema grammar")
    }

    func testParseForcedToolCallBareObject() {
        let c = ToolCalling.parseForcedToolCall(#"{"name":"get_weather","arguments":{"city":"Paris","days":3}}"#)
        XCTAssertEqual(c?.name, "get_weather")
        let args = parse(c!.argumentsJSON)
        XCTAssertEqual(args["city"] as? String, "Paris")
        XCTAssertEqual(args["days"] as? Int, 3)
    }

    func testParseForcedToolCallToleratesFenceAndWhitespace() {
        let c = ToolCalling.parseForcedToolCall("""

        ```json
        {"name": "add", "arguments": {"a": 1, "b": 2}}
        ```
        """)
        XCTAssertEqual(c?.name, "add")
        XCTAssertEqual(parse(c!.argumentsJSON)["a"] as? Int, 1)
    }

    func testParseForcedToolCallRejectsNonCall() {
        XCTAssertNil(ToolCalling.parseForcedToolCall("I cannot help with that."))
        XCTAssertNil(ToolCalling.parseForcedToolCall(#"{"foo":"bar"}"#))
    }

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
