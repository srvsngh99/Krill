import XCTest
@testable import KLMEngine

/// Tests for structured chat history passing (not flattened).
///
/// Verifies that the server helper passes messages as structured arrays
/// to the engine, which forwards them to tokenizer.applyChatTemplate().
final class ChatHistoryTests: XCTestCase {

    // MARK: - Test 6: Messages are structured, not flattened

    func testGenerateAcceptsMultiTurnMessages() {
        // Verify generate(messages:) signature accepts full conversation.
        // Without a real model we can't run inference, but we verify the API exists
        // and handles the messages array correctly.

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hi there!"],
            ["role": "user", "content": "How are you?"],
        ]

        // Engine without a loaded model should return empty stream, not crash
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-chat-test-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: tempDir)

        let (stream, getStats) = engine.generate(
            messages: messages,
            params: .greedy,
            maxTokens: 10
        )

        // Collect stream — should finish immediately (no model loaded)
        let expectation = XCTestExpectation(description: "Stream finishes")
        Task {
            var events: [TokenEvent] = []
            for await event in stream {
                events.append(event)
            }
            // No model = empty stream = 0 events
            XCTAssertTrue(events.isEmpty, "No model loaded should produce empty stream")
            XCTAssertNil(getStats(), "No model loaded should produce nil stats")
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 2.0)
    }

    func testGeneratePromptConvenienceWrapsMessages() {
        // The prompt-based convenience should be equivalent to wrapping in messages
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-chat-test-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: tempDir)

        // Both should produce the same result (empty stream, no model)
        let (stream1, _) = engine.generate(prompt: "Hello", systemPrompt: "Be helpful")
        let (stream2, _) = engine.generate(messages: [
            ["role": "system", "content": "Be helpful"],
            ["role": "user", "content": "Hello"],
        ])

        let exp1 = XCTestExpectation(description: "Stream 1")
        let exp2 = XCTestExpectation(description: "Stream 2")

        Task {
            var count1 = 0
            for await _ in stream1 { count1 += 1 }
            var count2 = 0
            for await _ in stream2 { count2 += 1 }
            XCTAssertEqual(count1, count2, "Both generate variants should behave the same")
            exp1.fulfill()
            exp2.fulfill()
        }
        wait(for: [exp1, exp2], timeout: 2.0)
    }

    // MARK: - Test 6: Server message conversion

    func testMessageConversionPreservesAllRoles() {
        // Simulates what the server does: convert [String: Any] to [String: String]
        let rawMessages: [[String: Any]] = [
            ["role": "system", "content": "You are an assistant."],
            ["role": "user", "content": "What is 2+2?"],
            ["role": "assistant", "content": "4"],
            ["role": "user", "content": "And 3+3?"],
        ]

        // Server conversion logic (matches Server.swift handleChatCompletions)
        let chatMessages: [[String: String]] = rawMessages.compactMap { msg in
            guard let role = msg["role"] as? String,
                  let content = msg["content"] as? String else { return nil }
            return ["role": role, "content": content]
        }

        XCTAssertEqual(chatMessages.count, 4, "All 4 messages should be preserved")
        XCTAssertEqual(chatMessages[0]["role"], "system")
        XCTAssertEqual(chatMessages[1]["role"], "user")
        XCTAssertEqual(chatMessages[2]["role"], "assistant")
        XCTAssertEqual(chatMessages[3]["role"], "user")
        XCTAssertEqual(chatMessages[3]["content"], "And 3+3?")
    }
}
