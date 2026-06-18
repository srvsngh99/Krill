import XCTest
@testable import KrillTokenizer

/// Detection of which chat templates expose a reasoning ("thinking") channel the
/// engine can turn on. Pure logic, no loaded tokenizer required.
final class ThinkingTemplateTests: XCTestCase {
    private func supports(external: String? = nil, embedded: String? = nil) -> Bool {
        KrillTokenizer.templateSupportsThinking(externalTemplate: external,
                                              embeddedTemplate: embedded)
    }

    func testGemmaChannelTemplateExternal() {
        // The Gemma-4 coder ships its channel template as an external .jinja.
        XCTAssertTrue(supports(external: "{{ '<|turn>model\\n' }}{% if enable_thinking %}..."))
        XCTAssertTrue(supports(external: "...<|channel>thought..."))
    }

    func testEmbeddedEnableThinkingTemplate() {
        // Qwen 3 etc. embed the template in tokenizer_config.json and branch on
        // enable_thinking.
        XCTAssertTrue(supports(embedded: "{% if enable_thinking %}<think>{% endif %}"))
    }

    func testExternalEnableThinkingTemplate() {
        XCTAssertTrue(supports(external: "{% if enable_thinking %}...{% endif %}"))
    }

    func testNonThinkingTemplatesAreNotDetected() {
        // A plain Llama/ChatML template with no thinking channel.
        XCTAssertFalse(supports(embedded: "<|im_start|>{{ message['role'] }}\\n{{ message['content'] }}<|im_end|>"))
        XCTAssertFalse(supports(external: "<start_of_turn>user\\n{{ content }}<end_of_turn>"))
        XCTAssertFalse(supports())  // no template at all
    }

    func testExternalPreferredOverEmbedded() {
        // External wins: a non-thinking external template hides an embedded
        // enable_thinking one (the external .jinja is what actually renders).
        XCTAssertFalse(supports(external: "<start_of_turn>user\\n{{ content }}",
                                embedded: "{% if enable_thinking %}<think>{% endif %}"))
    }
}
