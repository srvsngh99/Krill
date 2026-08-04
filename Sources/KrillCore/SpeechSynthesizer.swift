import Foundation
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// On-device text-to-speech for spoken model replies (voice phase 2): the chat
/// TUI can read each answer aloud, completing the hands-free loop (talk -> the
/// model answers in your ear). Wraps Apple's `AVSpeechSynthesizer`, which speaks
/// fully locally with the system voices - no download, no cloud. Mirrors
/// ``SpeechRecognizer`` (the dictation side).
public final class SpeechSynthesizer: @unchecked Sendable {
#if canImport(AVFoundation)
    // AVSpeechSynthesizer is main-thread-affine: it (and its utterances) must be
    // created and driven on the main thread, but the chat TUI run loop runs on a
    // cooperative-pool thread. So the synthesizer is created lazily on first use
    // inside `onMain` and every AVFoundation call hops to the main thread. It is
    // only ever read/written inside `onMain`, so the `@unchecked Sendable` mutable
    // state has no data race (the single serial caller is also serialized onto
    // main).
    private var synth: AVSpeechSynthesizer?
#endif
    private let settings: AppleSpeechSettings

    /// Empty identifiers use the system voice. An unavailable configured voice
    /// falls back to the configured language and then the system default.
    public init(language: String = "auto", voiceIdentifier: String = "", rate: Float = AppleSpeechSettings.systemRate) {
        self.settings = AppleSpeechSettings(language: language, voiceIdentifier: voiceIdentifier, rate: rate)
    }

    /// True when system text-to-speech is usable on this platform.
    public static var isAvailable: Bool {
#if canImport(AVFoundation)
        return true
#else
        return false
#endif
    }

    /// Speak `text` aloud, first cleaning it of markdown that reads badly out
    /// loud (code blocks, backticks, emphasis, headings). A new reply interrupts
    /// any in-flight utterance so speech tracks the latest answer. `rate` is the
    /// AVSpeech 0...1 rate (nil = the system default).
    public func speak(_ text: String, rate: Float? = nil) {
#if canImport(AVFoundation)
        speakChunks([text], rate: rate)
#endif
    }

    /// Clean and enqueue several completed sentence chunks as one reply. Unlike
    /// repeated `speak` calls, this interrupts the preceding reply exactly once,
    /// then lets AVSpeechSynthesizer naturally continue through each sentence.
    public func speakChunks(_ chunks: [String], rate: Float? = nil) {
#if canImport(AVFoundation)
        let cleanChunks = chunks.map(SpokenText.clean).filter { !$0.isEmpty }
        guard !cleanChunks.isEmpty else { return }
        onMain {
            let synth = self.synthesizer()
            synth.stopSpeaking(at: .immediate)
            let voice = Self.resolveVoice(settings: self.settings)
            let utteranceRate = rate ?? self.settings.utteranceRate
            for text in cleanChunks {
                let utterance = AVSpeechUtterance(string: text)
                utterance.voice = voice
                if let utteranceRate { utterance.rate = utteranceRate }
                synth.speak(utterance)
            }
        }
#endif
    }

    /// True while an utterance is being spoken.
    public var isSpeaking: Bool {
#if canImport(AVFoundation)
        return onMainSync { self.synth?.isSpeaking ?? false }
#else
        return false
#endif
    }

    /// Stop speaking immediately (e.g. the user cancelled the reply, started a
    /// new turn, or quit). Safe to call when nothing is speaking.
    public func stop() {
#if canImport(AVFoundation)
        onMain { self.synth?.stopSpeaking(at: .immediate) }
#endif
    }

#if canImport(AVFoundation)
    private static func resolveVoice(settings: AppleSpeechSettings) -> AVSpeechSynthesisVoice? {
        if let identifier = settings.requestedVoiceIdentifier,
           let voice = AVSpeechSynthesisVoice(identifier: identifier) {
            return voice
        }
        // If an identifier is stale (for example, a downloaded voice was
        // removed), `language` remains a useful, audible fallback.
        let requestedLanguage = settings.language.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requestedLanguage.isEmpty, requestedLanguage.lowercased() != "auto" else { return nil }
        return AVSpeechSynthesisVoice(language: settings.localeIdentifier)
    }

    /// Lazily create the synthesizer. MUST be called on the main thread (only
    /// invoked from inside `onMain`), satisfying AVSpeechSynthesizer's main-thread
    /// affinity.
    private func synthesizer() -> AVSpeechSynthesizer {
        if let synth { return synth }
        let created = AVSpeechSynthesizer()
        synth = created
        return created
    }

    /// Run an AVFoundation interaction on the main thread (synchronously if
    /// already there, else async-dispatched).
    private func onMain(_ body: @escaping @Sendable () -> Void) {
        if Thread.isMainThread {
            body()
        } else {
            DispatchQueue.main.async(execute: body)
        }
    }

    /// Read AVFoundation state from its owning thread while preserving the
    /// existing synchronous `isSpeaking` API for callers on any queue.
    private func onMainSync<T>(_ body: () -> T) -> T {
        if Thread.isMainThread { return body() }
        return DispatchQueue.main.sync(execute: body)
    }
#endif
}

/// Stateful sentence segmentation for streamed assistant text. `append` emits
/// only boundaries followed by whitespace (so a later closing quote stays with
/// its sentence); `finish` flushes the final remainder. The buffer is consumed
/// as it emits, so repeated partial transcripts cannot duplicate spoken text.
public struct SpeechSentenceChunker: Sendable {
    private var buffer = ""

    public init() {}

    public mutating func append(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        buffer += text
        return drainCompleteSentences()
    }

    public mutating func finish() -> [String] {
        defer { buffer = "" }
        let remaining = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        return remaining.isEmpty ? [] : [remaining]
    }

    public mutating func reset() { buffer = "" }

    private mutating func drainCompleteSentences() -> [String] {
        var chunks: [String] = []
        var searchStart = buffer.startIndex
        while searchStart < buffer.endIndex,
              let boundary = completeBoundary(from: searchStart) {
            let chunk = String(buffer[..<boundary]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty { chunks.append(chunk) }
            buffer.removeSubrange(..<boundary)
            while let first = buffer.first, first.isWhitespace { buffer.removeFirst() }
            searchStart = buffer.startIndex
        }
        return chunks
    }

    /// Return the index just after a complete sentence. Closing quotes/brackets
    /// immediately following punctuation are included in the spoken chunk.
    private func completeBoundary(from start: String.Index) -> String.Index? {
        var index = start
        while index < buffer.endIndex {
            let character = buffer[index]
            guard character == "." || character == "!" || character == "?" else {
                index = buffer.index(after: index)
                continue
            }
            var end = buffer.index(after: index)
            while end < buffer.endIndex, isClosingPunctuation(buffer[end]) {
                end = buffer.index(after: end)
            }
            // Waiting for whitespace prevents emitting `Hello.` too early when
            // the next streamed token is a closing quote.
            if end < buffer.endIndex, buffer[end].isWhitespace { return end }
            index = end
        }
        return nil
    }

    private func isClosingPunctuation(_ character: Character) -> Bool {
        switch character {
        case "\"", "'", "”", "’", ")", "]", "}": return true
        default: return false
        }
    }
}

/// Pure markdown-to-speech cleanup: strips the formatting that sounds wrong when
/// read aloud (fenced code, inline code backticks, `*`/`_` emphasis, heading and
/// list markers, link URLs) and collapses whitespace. Pure and unit-tested; the
/// synthesizer applies it before speaking.
public enum SpokenText {
    public static func clean(_ s: String) -> String {
        var out = s
        // Drop fenced code blocks entirely - reading code aloud is noise.
        out = replace(out, #"```[\s\S]*?```"#, with: " ")
        // Links: keep the visible text, drop the URL.
        out = replace(out, #"\[([^\]]+)\]\([^)]*\)"#, with: "$1")
        // Inline code: keep the contents, drop the backticks.
        out = replace(out, "`([^`]+)`", with: "$1")
        // Emphasis -> inner text. Each marker requires the content to start AND
        // end with a non-space char, and the `_` forms additionally require
        // non-word boundaries, so we do NOT swallow arithmetic (`2 * 3`) or
        // identifier underscores (`my_func_name`). Bold markers run before italic
        // so the doubled form is consumed first.
        // Strip ASTERISK emphasis only. The flanking classes include `*` itself,
        // so a marker adjacent to another asterisk (the `**` power operator, e.g.
        // `x**2 and y**3`) and arithmetic (`a*b`, `2 * 3`) are left intact.
        out = replace(out, #"(?<![A-Za-z0-9*])\*\*(\S(?:.*?\S)?)\*\*(?![A-Za-z0-9*])"#, with: "$1")  // **bold**
        out = replace(out, #"(?<![A-Za-z0-9*])\*(\S(?:.*?\S)?)\*(?![A-Za-z0-9*])"#, with: "$1")      // *italic*
        // UNDERSCORE emphasis (`_x_`, `__x__`) is deliberately NOT stripped: it is
        // structurally identical to the identifiers a coding model emits
        // constantly (snake_case, leading-underscore, and dunders like `__init__`
        // / `__main__` - including two dunders with text between them, which a
        // multi-word heuristic would wrongly span). No regex separates `__bold__`
        // from `__init__`, and underscore emphasis is rare in assistant prose
        // (everyone uses `*`/`**`), so we preserve identifiers and leave the rare
        // `_italic_` markers in place rather than corrupt code references.
        // Heading hashes and list bullets at the start of a line.
        out = replace(out, #"(?m)^\s{0,3}#{1,6}\s*"#, with: "")
        out = replace(out, #"(?m)^\s*[-*+]\s+"#, with: "")
        // Any leftover stray markers, then collapse whitespace.
        out = out.replacingOccurrences(of: "`", with: "")
        out = replace(out, #"\s+"#, with: " ")
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replace(_ s: String, _ pattern: String, with template: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
        let range = NSRange(s.startIndex..., in: s)
        return re.stringByReplacingMatches(in: s, range: range, withTemplate: template)
    }
}
