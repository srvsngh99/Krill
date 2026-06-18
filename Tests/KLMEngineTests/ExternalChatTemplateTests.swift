import Foundation
import XCTest
@testable import KLMTokenizer

/// Covers the file-load half of the external-`chat_template.jinja`
/// fix. The full tokenizer integration (decoding via swift-transformers'
/// Jinja engine, encoding back to ids) is exercised by the inference
/// smoke tests; this file isolates the on-disk discovery so a missing
/// file or a binary blob in the template path cannot regress silently.
final class ExternalChatTemplateTests: XCTestCase {

    func testReturnsTemplateContentsWhenFilePresent() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let jinja = """
        {%- for m in messages %}<|im_start|>{{m.role}}
        {{m.content}}<|im_end|>
        {%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """
        try jinja.write(
            to: dir.appendingPathComponent("chat_template.jinja"),
            atomically: true, encoding: .utf8)

        let loaded = KLMTokenizer.readExternalChatTemplate(directory: dir)
        XCTAssertEqual(loaded, jinja)
    }

    func testReturnsNilWhenFileAbsent() throws {
        // Older HF checkpoints (Qwen 2.5 14B, Llama 3.x) embed the
        // template inside `tokenizer_config.json` and ship no
        // `chat_template.jinja`. The loader must report `nil` so the
        // wrapper falls back to the embedded-template path; if it
        // returned an empty string or an error the chat path would
        // break for every pre-2025 checkpoint.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(KLMTokenizer.readExternalChatTemplate(directory: dir))
    }

    func testReturnsNilWhenFileEmpty() throws {
        // An empty `chat_template.jinja` cannot drive the Jinja engine.
        // Treat it as absent so the fallback chain (embedded ->
        // Gemma 4 manual -> Llama 3 manual) gets a chance to apply.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data().write(to: dir.appendingPathComponent("chat_template.jinja"))
        XCTAssertNil(KLMTokenizer.readExternalChatTemplate(directory: dir))
    }

    func testReturnsNilWhenFileNotUTF8() throws {
        // Defensive: a non-UTF-8 blob at the template path should
        // produce nil rather than crashing the load.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 0xFF 0xFE is a UTF-16 BOM, not a valid UTF-8 start.
        let garbage = Data([0xFF, 0xFE, 0xFF, 0xFE])
        try garbage.write(to: dir.appendingPathComponent("chat_template.jinja"))
        XCTAssertNil(KLMTokenizer.readExternalChatTemplate(directory: dir))
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-tpl-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }
}
