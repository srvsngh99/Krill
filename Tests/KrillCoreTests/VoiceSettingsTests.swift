import XCTest
@testable import KrillCore

final class VoiceSettingsTests: XCTestCase {
    func testAutoLanguageUsesCurrentLocaleAndSystemRate() {
        let settings = AppleSpeechSettings(language: "", voiceIdentifier: "", rate: 0)
        XCTAssertEqual(settings.localeIdentifier, Locale.current.identifier)
        XCTAssertNil(settings.requestedVoiceIdentifier)
        XCTAssertNil(settings.utteranceRate)
    }

    func testExplicitLanguageVoiceAndRateArePreserved() {
        let settings = AppleSpeechSettings(language: "en-GB", voiceIdentifier: "com.example.voice", rate: 0.47)
        XCTAssertEqual(settings.localeIdentifier, "en-GB")
        XCTAssertEqual(settings.requestedVoiceIdentifier, "com.example.voice")
        XCTAssertEqual(settings.utteranceRate, 0.47)
    }

    func testInvalidRatesSafelyUseSystemDefault() {
        XCTAssertNil(AppleSpeechSettings(rate: -1).utteranceRate)
        XCTAssertNil(AppleSpeechSettings(rate: 1.1).utteranceRate)
    }
}
