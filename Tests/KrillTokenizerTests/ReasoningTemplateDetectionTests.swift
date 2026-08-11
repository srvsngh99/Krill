import XCTest
@testable import KrillTokenizer

/// `emitsReasoningBlock` decides the CLI's output-token budget, and getting it
/// wrong is user-visible in the worst way: a thinking model whose template opens
/// `<think>` spends the whole budget on hidden reasoning, so `krill run <model>
/// "hello"` prints NOTHING. That reads as a broken model, not a truncated answer
/// — Nanbeige 4.2 shipped exactly this failure before the budget became
/// reasoning-aware.
///
/// The markers deliberately mirror `StreamingReasoningFilter`'s opening
/// sentinels, so "the template opens a block" and "the output gets suppressed"
/// cannot drift apart.
final class ReasoningTemplateDetectionTests: XCTestCase {

    private func detects(_ template: String?) -> Bool {
        KrillTokenizer.templateEmitsReasoningBlock(template)
    }

    func testDetectsThinkTag() {
        // Nanbeige 4.2: the template literally ends the prompt with `<think>\n`.
        XCTAssertTrue(detects("<|im_start|>assistant\n<think>\n"))
    }

    func testDetectsEnableThinkingFlag() {
        // Qwen 3 / Nanbeige gate the block on a flag that defaults ON, so the
        // flag's presence is itself the signal.
        XCTAssertTrue(detects("{%- if enable_thinking is defined %}...{%- endif %}"))
    }

    func testDetectsGemmaChannelAndVariants() {
        XCTAssertTrue(detects("<|channel>thought"))
        XCTAssertTrue(detects("<thinking>"))
        XCTAssertTrue(detects("<|think|>"))
    }

    func testPlainChatMLIsNotReasoning() {
        // A non-thinking model must KEEP the lean single-shot budget; widening it
        // for everyone would make every short prompt pay for headroom it cannot use.
        XCTAssertFalse(detects(
            "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>assistant\n"))
    }

    func testLlamaStyleTemplateIsNotReasoning() {
        XCTAssertFalse(detects(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"))
    }

    func testNilTemplateIsNotReasoning() {
        XCTAssertFalse(detects(nil))
    }

    func testEmptyTemplateIsNotReasoning() {
        XCTAssertFalse(detects(""))
    }

    /// The directory reader is what the `krill run` daemon route uses: it must
    /// answer without constructing a tokenizer or loading weights.
    func testDirectoryReaderFindsEmbeddedTemplate() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-reasoning-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = ["chat_template": "<|im_start|>assistant\n<think>\n"]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))

        XCTAssertTrue(KrillTokenizer.emitsReasoningBlock(inDirectory: dir))
    }

    /// Muse Glimmer fences reasoning by RECIPIENT, not by tag: the template
    /// renders a prior turn's `reasoning_content` as `to=self`. Missing this
    /// starves the model of headroom and the reply comes back empty.
    func testDetectsAtemRecipientChannel() {
        let template = "{%- if message.get('reasoning_content') -%}"
            + "{{- '<|start|>assistant to=self<|message|>' + message['reasoning_content'] }}"
        XCTAssertTrue(KrillTokenizer.templateEmitsReasoningBlock(template))
    }

    func testDirectoryReaderReturnsFalseWhenNoTemplate() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-reasoning-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertFalse(KrillTokenizer.emitsReasoningBlock(inDirectory: dir))
    }
}
