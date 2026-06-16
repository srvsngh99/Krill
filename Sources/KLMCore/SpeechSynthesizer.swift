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
    private let synth = AVSpeechSynthesizer()
#endif
    public init() {}

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
        let clean = SpokenText.clean(text)
        guard !clean.isEmpty else { return }
        synth.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: clean)
        if let rate { utterance.rate = rate }
        synth.speak(utterance)
#endif
    }

    /// True while an utterance is being spoken.
    public var isSpeaking: Bool {
#if canImport(AVFoundation)
        return synth.isSpeaking
#else
        return false
#endif
    }

    /// Stop speaking immediately (e.g. the user cancelled the reply or started a
    /// new turn). Safe to call when nothing is speaking.
    public func stop() {
#if canImport(AVFoundation)
        synth.stopSpeaking(at: .immediate)
#endif
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
        // Emphasis: **bold**, *italic*, __bold__, _italic_ -> the inner text.
        out = replace(out, #"(\*\*|\*|__|_)(.+?)\1"#, with: "$2")
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
