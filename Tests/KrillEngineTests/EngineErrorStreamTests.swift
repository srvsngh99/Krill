import XCTest
@testable import KrillEngine

/// PR #21 rereview P1b regression: a hard pre-generation failure (native
/// audio decode error) must not be swallowed by consumers that terminate on
/// `isEnd` before reading `text`. This pins the consumer-safe shape of
/// `InferenceEngine.mediaErrorStream` without needing a loaded model.
final class EngineErrorStreamTests: XCTestCase {

    private func collect(_ s: AsyncStream<TokenEvent>) async -> [TokenEvent] {
        var out: [TokenEvent] = []
        for await ev in s { out.append(ev) }
        return out
    }

    func testErrorMessageIsANonTerminalEventFollowedByTerminalEnd() async {
        let msg = "Error: native audio decode failed: bad WAV"
        let (stream, stats) = InferenceEngine.mediaErrorStream(msg)
        let events = await collect(stream)

        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0].text, msg)
        XCTAssertFalse(events[0].isEnd, "message must be a NON-terminal event")
        XCTAssertTrue(events[1].isEnd, "a separate terminal event must follow")
        XCTAssertTrue(events[1].text.isEmpty)
        XCTAssertNil(stats())
    }

    /// The two real-world consumer idioms in the codebase must both end up
    /// with the error text (CLI / non-streaming break on `isEnd`; streaming
    /// appends every chunk).
    func testBothConsumerIdiomsSurfaceTheError() async {
        let msg = "Error: native audio decode failed: corrupt RIFF"

        // Idiom A: break on isEnd BEFORE appending (CLI, non-streaming loops).
        let (sA, _) = InferenceEngine.mediaErrorStream(msg)
        var a = ""
        for await ev in sA { if ev.isEnd { break }; a += ev.text }
        XCTAssertTrue(a.contains("native audio decode failed"),
                      "break-on-isEnd consumer lost the error: \(a)")

        // Idiom B: append THEN break (streaming-style accumulation).
        let (sB, _) = InferenceEngine.mediaErrorStream(msg)
        var b = ""
        for await ev in sB { b += ev.text; if ev.isEnd { break } }
        XCTAssertTrue(b.contains("native audio decode failed"),
                      "append-then-break consumer lost the error: \(b)")
    }
}
