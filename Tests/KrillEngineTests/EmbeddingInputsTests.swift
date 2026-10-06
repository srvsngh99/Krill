import XCTest
import Foundation
@testable import KrillEngine

/// Request parsing for `/v1/embeddings` / `/api/embed` content parts.
/// Pure: no model, no server.
final class EmbeddingInputsTests: XCTestCase {
    private let tinyPNG =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="

    private func json(_ s: String) -> Any {
        try! JSONSerialization.jsonObject(with: Data(s.utf8), options: [.fragmentsAllowed])
    }

    func testPlainStringsAreUnchanged() throws {
        XCTAssertEqual(try EmbeddingInputParser.parse("hello"), [EmbeddingInput(text: "hello")])
        let many = try EmbeddingInputParser.parse(json(#"["a", "b", ""]"#))
        XCTAssertEqual(many.map { $0.joinedText }, ["a", "b", ""])
        XCTAssertTrue(many.allSatisfy { !$0.hasMedia })
    }

    func testBadShapesAreClientErrors() {
        for bad in ["[]", "[1,2,3]", "[[1,2]]", "{}", "42", "null", #"[{"nope":1}]"#, #"[{"content":[]}]"#] {
            XCTAssertThrowsError(try EmbeddingInputParser.parse(json(bad)), bad) { e in
                XCTAssertTrue(e is EmbeddingInputError, bad)
            }
        }
    }

    func testTextAndImageParts() throws {
        let body = """
        ["plain", {"content": [{"type":"text","text":"a red bicycle"},
                               {"type":"image_url","image_url":{"url":"data:image/png;base64,\(tinyPNG)"}}]}]
        """
        let r = try EmbeddingInputParser.parse(json(body))
        XCTAssertEqual(r.count, 2)
        XCTAssertFalse(r[0].hasMedia)
        XCTAssertTrue(r[1].hasMedia)
        guard case .text(let t) = r[1].parts[0], case .image(let d) = r[1].parts[1] else {
            return XCTFail("wrong part kinds: \(r[1].parts)")
        }
        XCTAssertEqual(t, "a red bicycle")
        XCTAssertEqual(d.prefix(4), Data([0x89, 0x50, 0x4E, 0x47]), "decoded to PNG magic")
    }

    func testPartOrderIsPreserved() throws {
        let body = """
        [{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,\(tinyPNG)"}},
                     {"type":"text","text":"x"},
                     {"type":"input_image","data":"\(tinyPNG)"},
                     {"type":"text","text":"y"}]}]
        """
        let item = try EmbeddingInputParser.parse(json(body))[0]
        let kinds = item.parts.map { p -> String in
            switch p { case .text(let t): return "t:" + t; case .image: return "img" }
        }
        XCTAssertEqual(kinds, ["img", "t:x", "img", "t:y"])
    }

    func testImageUrlAcceptsBareStringAndInputImageVariants() throws {
        let uri = "data:image/jpeg;base64,\(tinyPNG)"
        for part in [
            #"{"type":"image_url","image_url":"\#(uri)"}"#,
            #"{"type":"input_image","image_url":"\#(uri)"}"#,
            #"{"type":"input_image","image_url":{"url":"\#(uri)"}}"#,
            #"{"type":"input_image","data":"\#(tinyPNG)"}"#,
        ] {
            let r = try EmbeddingInputParser.parse(json("[{\"content\":[\(part)]}]"))
            XCTAssertEqual(r.count, 1, part)
            guard case .image = r[0].parts[0] else { return XCTFail(part) }
        }
        // a single bare part and a bare array of parts are single items
        XCTAssertEqual(try EmbeddingInputParser.parse(json(#"[{"type":"text","text":"hi"}]"#)).count, 1)
        XCTAssertEqual(try EmbeddingInputParser.parse(
            json(#"[[{"type":"text","text":"a"},{"type":"text","text":"b"}]]"#))[0].joinedText, "ab")
    }

    func testRemoteUrlsAreRefused() {
        for url in ["http://x.test/a.png", "https://x.test/a.png", "file:///etc/passwd", "/etc/passwd"] {
            let body = #"[{"content":[{"type":"image_url","image_url":{"url":"\#(url)"}}]}]"#
            XCTAssertThrowsError(try EmbeddingInputParser.parse(json(body)), url) {
                XCTAssertTrue(("\($0)").contains("not fetched"), "\($0)")
            }
        }
    }

    func testMalformedMedia() {
        let bad = [
            #"{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!notbase64###"}}"#,
            #"{"type":"image_url","image_url":{"url":"data:image/png,rawbytes"}}"#,
            #"{"type":"image_url","image_url":{"url":"data:audio/wav;base64,\#(tinyPNG)"}}"#,
            #"{"type":"image_url"}"#,
            #"{"type":"image_url","image_url":{"url":""}}"#,
            #"{"type":"text"}"#,
            #"{"text":"no type"}"#,
        ]
        for part in bad {
            XCTAssertThrowsError(try EmbeddingInputParser.parse(json("[{\"content\":[\(part)]}]")), part) {
                XCTAssertTrue($0 is EmbeddingInputError)
            }
        }
    }

    func testAudioAndVideoPartsAreRecognisedButNotYetSupported() {
        for (type, body) in [
            ("input_audio", #"{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}"#),
            ("video_url", #"{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}"#),
            ("input_video", #"{"type":"input_video","data":"AAAA"}"#),
        ] {
            XCTAssertThrowsError(try EmbeddingInputParser.parse(json("[{\"content\":[\(body)]}]")), type) {
                XCTAssertTrue("\($0)".contains("not yet supported"), "\($0)")
            }
        }
        XCTAssertThrowsError(try EmbeddingInputParser.parse(json(#"[{"content":[{"type":"hologram"}]}]"#))) {
            XCTAssertTrue("\($0)".contains("unknown part type"), "\($0)")
        }
    }

    func testTextOnlyContentItemsTakeTheTextPath() throws {
        let r = try EmbeddingInputParser.parse(
            json(#"[{"content":[{"type":"text","text":"foo "},{"type":"text","text":"bar"}]}, {"content":"str"}]"#))
        XCTAssertFalse(r[0].hasMedia); XCTAssertEqual(r[0].joinedText, "foo bar")
        XCTAssertFalse(r[1].hasMedia); XCTAssertEqual(r[1].joinedText, "str")
    }
}
