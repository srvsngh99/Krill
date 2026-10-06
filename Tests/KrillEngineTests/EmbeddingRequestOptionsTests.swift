import XCTest
import Foundation
@testable import KrillEngine

/// Request-field parsing and prompt selection for the embeddings endpoints.
/// No model or weights needed.
final class EmbeddingRequestOptionsTests: XCTestCase {

    func testNoOptionsIsDefault() throws {
        XCTAssertEqual(try EmbeddingRequestOptions.parse(["model": "m", "input": "x"]),
                       EmbeddingRequestOptions())
    }

    func testParsesDimensionsTaskInstruction() throws {
        let a = try EmbeddingRequestOptions.parse(["dimensions": 256, "task": "SearchQuery"])
        XCTAssertEqual(a.dimensions, 256)
        XCTAssertEqual(a.task, "SearchQuery")
        XCTAssertNil(a.instruction)
        let b = try EmbeddingRequestOptions.parse(["instruction": "Instruct: q\nQuery: "])
        XCTAssertEqual(b.instruction, "Instruct: q\nQuery: ")
    }

    func testParsesFromRealJSONBody() throws {
        let body = #"{"model":"m","input":["a"],"dimensions":128,"task":"Document"}"#
        let json = try JSONSerialization.jsonObject(with: Data(body.utf8)) as! [String: Any]
        let o = try EmbeddingRequestOptions.parse(json)
        XCTAssertEqual(o.dimensions, 128)
        XCTAssertEqual(o.task, "Document")
    }

    func testRejectsBadDimensions() {
        for bad: Any in [0, -4, 2.5, "256", true, [1]] {
            XCTAssertThrowsError(try EmbeddingRequestOptions.parse(["dimensions": bad]),
                                 "dimensions \(bad) should be rejected")
        }
        // explicit null is "absent"
        XCTAssertNoThrow(try EmbeddingRequestOptions.parse(["dimensions": NSNull()]))
    }

    func testRejectsBadTaskAndTaskPlusInstruction() {
        XCTAssertThrowsError(try EmbeddingRequestOptions.parse(["task": ""]))
        XCTAssertThrowsError(try EmbeddingRequestOptions.parse(["task": 3]))
        XCTAssertThrowsError(
            try EmbeddingRequestOptions.parse(["task": "Document", "instruction": "x"]))
    }

    // MARK: - Prompt table (values copied from the model's config_sentence_transformers.json)

    private let table = EmbeddingPromptTable(prompts: [
        "SearchQuery": "task: search result | query: ",
        "Document": "title: none | text: ",
        "Classification": "task: classification | query: ",
        "document": "title: none | text: ",
    ])

    func testPromptPrefixExactThenCaseInsensitive() {
        XCTAssertEqual(table.prefix(for: "SearchQuery"), "task: search result | query: ")
        XCTAssertEqual(table.prefix(for: "Document"), "title: none | text: ")
        XCTAssertEqual(table.prefix(for: "searchquery"), "task: search result | query: ")
        XCTAssertNil(table.prefix(for: "Nope"))
        XCTAssertTrue(table.taskNames.contains("Classification"))
    }

    func testPromptApplicationIsPlainConcatenation() {
        // sentence-transformers does prompt + text before tokenisation.
        let text = "What causes the northern lights?"
        XCTAssertEqual((table.prefix(for: "SearchQuery") ?? "") + text,
                       "task: search result | query: What causes the northern lights?")
    }

    func testPromptTableLoadsFromDirectory() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(EmbeddingPromptTable.load(directory: dir))
        let json = #"{"prompts": {"query": "q: ", "document": "d: "}, "default_prompt_name": null}"#
        try json.write(to: dir.appendingPathComponent("config_sentence_transformers.json"),
                       atomically: true, encoding: .utf8)
        XCTAssertEqual(EmbeddingPromptTable.load(directory: dir)?.prefix(for: "query"), "q: ")
        // an empty prompts table (many ST models ship `"prompts": {}`) is "no table"
        try #"{"prompts": {}}"#.write(to: dir.appendingPathComponent("config_sentence_transformers.json"),
                                      atomically: true, encoding: .utf8)
        XCTAssertNil(EmbeddingPromptTable.load(directory: dir))
    }
}
