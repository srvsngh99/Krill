import XCTest
@testable import KrillCore

/// Unit tests for the interactive media-attachment helpers: magic-byte / ext
/// kind detection, terminal path normalization (drag-drop, quotes, `~`), and
/// WAV encoding (used by live mic capture). Deterministic, no external assets.
final class MediaAttachmentTests: XCTestCase {

    // MARK: - detectKind via magic bytes

    func testDetectsImageMagicBytes() {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0])
        let jpg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0, 0])
        let gif = Data(Array("GIF89a".utf8) + [0, 0])
        var webp = Data(Array("RIFF".utf8)); webp.append(Data([0, 0, 0, 0])); webp.append(Data(Array("WEBP".utf8)))
        XCTAssertEqual(MediaAttachment.detectKind(data: png, pathExtension: ""), .image)
        XCTAssertEqual(MediaAttachment.detectKind(data: jpg, pathExtension: ""), .image)
        XCTAssertEqual(MediaAttachment.detectKind(data: gif, pathExtension: ""), .image)
        XCTAssertEqual(MediaAttachment.detectKind(data: webp, pathExtension: ""), .image)
    }

    func testDetectsAudioMagicBytes() {
        var wav = Data(Array("RIFF".utf8)); wav.append(Data([0, 0, 0, 0])); wav.append(Data(Array("WAVE".utf8)))
        let flac = Data(Array("fLaC".utf8) + [0, 0])
        let ogg = Data(Array("OggS".utf8) + [0, 0])
        let id3 = Data(Array("ID3".utf8) + [0, 0, 0])
        let m4a = Data([0, 0, 0, 0x18] + Array("ftyp".utf8) + Array("M4A ".utf8))
        XCTAssertEqual(MediaAttachment.detectKind(data: wav, pathExtension: ""), .audio)
        XCTAssertEqual(MediaAttachment.detectKind(data: flac, pathExtension: ""), .audio)
        XCTAssertEqual(MediaAttachment.detectKind(data: ogg, pathExtension: ""), .audio)
        XCTAssertEqual(MediaAttachment.detectKind(data: id3, pathExtension: ""), .audio)
        XCTAssertEqual(MediaAttachment.detectKind(data: m4a, pathExtension: ""), .audio)
    }

    /// A RIFF/WAVE header must classify as audio, never as a WebP image.
    func testWavNotMisreadAsImage() {
        var wav = Data(Array("RIFF".utf8)); wav.append(Data([0, 0, 0, 0])); wav.append(Data(Array("WAVE".utf8)))
        XCTAssertFalse(MediaAttachment.isImageData(wav))
        XCTAssertTrue(MediaAttachment.isAudioData(wav))
    }

    // MARK: - detectKind via extension fallback

    func testExtensionFallbackWhenBytesUnknown() {
        let junk = Data([0x12, 0x34, 0x56, 0x78, 0x9A])
        XCTAssertEqual(MediaAttachment.detectKind(data: junk, pathExtension: "png"), .image)
        XCTAssertEqual(MediaAttachment.detectKind(data: junk, pathExtension: "JPG"), .image)
        XCTAssertEqual(MediaAttachment.detectKind(data: junk, pathExtension: "wav"), .audio)
        XCTAssertEqual(MediaAttachment.detectKind(data: junk, pathExtension: "m4a"), .audio)
        XCTAssertNil(MediaAttachment.detectKind(data: junk, pathExtension: "txt"))
        XCTAssertNil(MediaAttachment.detectKind(data: junk, pathExtension: ""))
    }

    // MARK: - normalizePath

    func testNormalizeStripsQuotes() {
        XCTAssertEqual(MediaAttachment.normalizePath("\"/tmp/a b.png\""), "/tmp/a b.png")
        XCTAssertEqual(MediaAttachment.normalizePath("'/tmp/a b.png'"), "/tmp/a b.png")
    }

    func testNormalizeUnescapesDraggedPath() {
        // Terminal drag-and-drop escapes spaces and parens with backslashes.
        XCTAssertEqual(MediaAttachment.normalizePath("/Users/me/My\\ Photos/cat\\ \\(1\\).png"),
                       "/Users/me/My Photos/cat (1).png")
    }

    func testNormalizeExpandsTilde() {
        XCTAssertEqual(MediaAttachment.normalizePath("~"), NSHomeDirectory())
        XCTAssertEqual(MediaAttachment.normalizePath("~/Pictures/cat.png"),
                       NSHomeDirectory() + "/Pictures/cat.png")
    }

    func testNormalizePlainPathUnchanged() {
        XCTAssertEqual(MediaAttachment.normalizePath("  /tmp/plain.png  "), "/tmp/plain.png")
    }

    // MARK: - prompt media-reference extraction

    func testExtractsBarePathMidSentence() {
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(
            from: "What is in this screenshot? /tmp/shot.png please explain."
        ) { token in
            token == "/tmp/shot.png" ? .attach(token) : .unresolved
        }
        XCTAssertEqual(result.cleaned, "What is in this screenshot?  please explain.")
        XCTAssertEqual(result.attachments, ["/tmp/shot.png"])
        XCTAssertTrue(result.unsupported.isEmpty)
    }

    func testExtractsMarkedPathMidSentence() {
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(
            from: "Compare this @/tmp/shot.png with the mockup."
        ) { token in
            token == "/tmp/shot.png" ? .attach(token) : .unresolved
        }
        XCTAssertEqual(result.cleaned, "Compare this  with the mockup.")
        XCTAssertEqual(result.attachments, ["/tmp/shot.png"])
    }

    func testExtractsEscapedBarePathAsOneToken() {
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(
            from: "Inspect /tmp/Screen\\ Shot\\ \\(final\\).png now"
        ) { token in
            token == "/tmp/Screen\\ Shot\\ \\(final\\).png" ? .attach(token) : .unresolved
        }
        XCTAssertEqual(result.cleaned, "Inspect  now")
        XCTAssertEqual(result.attachments, ["/tmp/Screen\\ Shot\\ \\(final\\).png"])
    }

    func testExtractsWholeBarePath() {
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(
            from: "~/Pictures/shot.png"
        ) { token in
            token == "~/Pictures/shot.png" ? .attach(token) : .unresolved
        }
        XCTAssertEqual(result.cleaned, "")
        XCTAssertEqual(result.attachments, ["~/Pictures/shot.png"])
    }

    func testExtractsDotRelativePathMidSentence() {
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(
            from: "Use ./Fixtures/shot.png for the comparison"
        ) { token in
            token == "./Fixtures/shot.png" ? .attach(token) : .unresolved
        }
        XCTAssertEqual(result.cleaned, "Use  for the comparison")
        XCTAssertEqual(result.attachments, ["./Fixtures/shot.png"])
    }

    func testLeavesUnresolvedCandidatesVerbatim() {
        var resolved: [String] = []
        let input = "Inspect /tmp/missing.png and @~/missing.wav"
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(from: input) { token in
            resolved.append(token)
            return .unresolved
        }
        XCTAssertEqual(result.cleaned, input)
        XCTAssertTrue(result.attachments.isEmpty)
        XCTAssertTrue(result.unsupported.isEmpty)
        XCTAssertEqual(resolved, ["/tmp/missing.png", "~/missing.wav"])
    }

    func testProseNeverInvokesResolver() {
        var resolved: [String] = []
        let input = "The value is 3.5 and/or thereabouts."
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(from: input) { token in
            resolved.append(token)
            return .unresolved
        }
        XCTAssertEqual(result.cleaned, input)
        XCTAssertTrue(resolved.isEmpty)
    }

    func testUnsupportedReferenceProducesNoteMetadataAndStaysInPrompt() {
        let input = "Please transcribe /tmp/voice.wav"
        let result: MediaReferenceExtraction<String> = MediaAttachment.extractReferences(from: input) { token in
            token == "/tmp/voice.wav" ? .unsupported(.audio) : .unresolved
        }
        XCTAssertEqual(result.cleaned, input)
        XCTAssertTrue(result.attachments.isEmpty)
        XCTAssertEqual(result.unsupported, [UnsupportedMediaReference(token: "/tmp/voice.wav", kind: .audio)])
    }

    func testDetectsPNGFromSixtyFourByteHeader() {
        var header = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        header.append(Data(repeating: 0, count: 56))
        XCTAssertEqual(header.count, 64)
        XCTAssertEqual(MediaAttachment.detectKind(data: header, pathExtension: ""), .image)
    }

    // MARK: - imageDimensions

    func testPNGDimensions() {
        // 5x3 PNG IHDR: 8-byte sig, "IHDR" len, then width=5,height=3 big-endian.
        var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        png.append(Data([0, 0, 0, 13]))                   // IHDR length
        png.append(Data(Array("IHDR".utf8)))
        png.append(Data([0, 0, 0, 5, 0, 0, 0, 3]))        // width=5, height=3
        png.append(Data([8, 2, 0, 0, 0]))                 // bit depth, color type, ...
        let dims = MediaAttachment.imageDimensions(png)
        XCTAssertEqual(dims?.width, 5)
        XCTAssertEqual(dims?.height, 3)
    }

    func testGIFDimensions() {
        // "GIF89a" then width=4 (LE), height=7 (LE).
        var gif = Data(Array("GIF89a".utf8))
        gif.append(Data([4, 0, 7, 0]))
        let dims = MediaAttachment.imageDimensions(gif)
        XCTAssertEqual(dims?.width, 4)
        XCTAssertEqual(dims?.height, 7)
    }

    func testUnknownImageDimensionsNil() {
        XCTAssertNil(MediaAttachment.imageDimensions(Data([0x12, 0x34, 0x56])))
    }

    // MARK: - encodeWAV

    func testEncodeWAVHeaderAndRoundtrip() throws {
        // A short ramp at 16 kHz; encodeWAV must produce a valid WAV that the
        // existing AudioPreprocessor decodes back to (approximately) the input.
        let n = 320
        let samples = (0..<n).map { Float($0) / Float(n) * 2 - 1 }   // -1 .. ~1
        let wav = MediaAttachment.encodeWAV(samples: samples, sampleRate: 16_000)

        XCTAssertTrue(AudioPreprocessor.isWAV(wav), "encodeWAV must emit a RIFF/WAVE header")
        XCTAssertEqual(MediaAttachment.detectKind(data: wav, pathExtension: ""), .audio)

        let decoded = try AudioPreprocessor.monoWaveform(fromAudio: wav)
        XCTAssertEqual(decoded.count, n)
        for (a, b) in zip(samples, decoded) {
            XCTAssertEqual(a, b, accuracy: 1.0 / 32767.0 + 1e-4, "16-bit quantization roundtrip")
        }
    }
}
