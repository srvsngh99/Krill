import XCTest
@testable import KrillCore

final class VoiceSessionTests: XCTestCase {
    private let sampleRate = 1_000.0

    func testSignalLevelMeasuresRMSAndSilence() {
        let level = VoiceSignalLevel.measure([1, -1, 1, -1])
        XCTAssertEqual(level.rms, 1, accuracy: 0.0001)
        XCTAssertEqual(level.decibels, 0, accuracy: 0.0001)
        XCTAssertEqual(VoiceSignalLevel.measure([]).decibels, -160)
    }

    func testEndpointingRequiresConsecutiveSpeechAndRetainsPreRoll() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.08, preRoll: 0.05, trailing: 0.06, minimum: 0.08))
        XCTAssertTrue(detector.process(frame(0, count: 50)).isEmpty)
        XCTAssertTrue(detector.process(frame(0.1, count: 40)).isEmpty)

        let start = detector.process(frame(0.1, count: 40))
        guard let startEvent = start.first,
              case let .speechStarted(level, audio) = startEvent else {
            return XCTFail("expected confirmed speech start")
        }
        XCTAssertGreaterThan(level.decibels, -38)
        // 50 ms pre-roll + two consecutive 40 ms confirmation frames.
        XCTAssertEqual(audio.samples.count, 130)
        XCTAssertEqual(audio.sampleRate, sampleRate)

        let continued = detector.process(frame(0.1, count: 20))
        guard let continuedEvent = continued.first,
              case let .speechContinued(_, audio) = continuedEvent else {
            return XCTFail("expected continued speech")
        }
        XCTAssertEqual(audio.samples.count, 20)
    }

    func testEndpointingUsesHysteresisAndTrailingSilence() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.02, preRoll: 0, trailing: 0.06, minimum: 0.02))
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        // -40 dBFS is below start (-38) but above continuation (-45), keeping
        // an established utterance active rather than prematurely ending it.
        XCTAssertEqual(detector.process(frame(0.01, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 30)).count, 1)
        let result = detector.process(frame(0, count: 30))
        guard let finalEvent = result.first,
              case let .utteranceReady(samples, rate) = finalEvent else {
            return XCTFail("expected silence endpoint")
        }
        XCTAssertEqual(rate, sampleRate)
        XCTAssertEqual(samples.count, 100)
    }

    func testEndpointingDropsTooShortSpeechAndCanBeReused() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.02, preRoll: 0, trailing: 0.03, minimum: 0.08))
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 20)).count, 1)
        XCTAssertTrue(detector.process(frame(0, count: 20)).isEmpty)

        // The rejected blip did not poison the next genuine instruction.
        XCTAssertEqual(detector.process(frame(0.1, count: 80)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 20)).count, 1)
    }

    func testEndpointingCapsMaximumUtterance() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.02, preRoll: 0, trailing: 1, minimum: 0.02, maximum: 0.10))
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0.1, count: 40)).count, 1)
        let result = detector.process(frame(0.1, count: 50))
        guard let finalEvent = result.first,
              case let .utteranceReady(samples, _) = finalEvent else {
            return XCTFail("expected maximum duration endpoint")
        }
        XCTAssertEqual(samples.count, 100)
    }

    func testSentenceChunkerKeepsQuotesAndNeverRepeatsEmission() {
        var chunker = SpeechSentenceChunker()
        XCTAssertEqual(chunker.append("Ship it."), [])
        XCTAssertEqual(chunker.append("\" Next idea"), ["Ship it.\""])
        XCTAssertEqual(chunker.append(" is here! "), ["Next idea is here!"])
        XCTAssertEqual(chunker.append("Last one"), [])
        XCTAssertEqual(chunker.finish(), ["Last one"])
        XCTAssertEqual(chunker.finish(), [])
    }

    func testSentenceChunkerHandlesSeveralBoundariesAndClosingBrackets() {
        var chunker = SpeechSentenceChunker()
        XCTAssertEqual(chunker.append("One. Two? Three!) Four"), ["One.", "Two?", "Three!)"])
        XCTAssertEqual(chunker.finish(), ["Four"])
    }

    private func configuration(
        confirmation: TimeInterval,
        preRoll: TimeInterval,
        trailing: TimeInterval,
        minimum: TimeInterval,
        maximum: TimeInterval = 2
    ) -> VoiceEndpointingConfiguration {
        .init(
            speechThresholdDecibels: -38,
            silenceThresholdDecibels: -45,
            speechConfirmationDuration: confirmation,
            preRollDuration: preRoll,
            minimumSpeechDuration: minimum,
            trailingSilenceDuration: trailing,
            maximumUtteranceDuration: maximum)
    }

    private func frame(_ amplitude: Float, count: Int) -> VoiceAudioFrame {
        .init(samples: Array(repeating: amplitude, count: count), sampleRate: sampleRate)
    }
}
