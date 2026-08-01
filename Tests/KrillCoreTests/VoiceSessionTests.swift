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
        XCTAssertEqual(detector.process(frame(0, count: 20)), [.utteranceDiscarded])

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

    func testEndpointingRetainsRecentAudioAcrossUtterances() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.02, preRoll: 0.02, trailing: 0.02, minimum: 0.02))
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 20)).count, 1)

        let next = detector.process(frame(0.1, count: 20))
        guard let event = next.first,
              case let .speechStarted(_, audio) = event else {
            return XCTFail("expected the next utterance to start")
        }
        XCTAssertEqual(audio.samples.count, 40)
        XCTAssertEqual(Array(audio.samples.prefix(20)), Array(repeating: 0, count: 20))
    }

    func testMaximumDurationSplitDoesNotReplaySpeechAsPreRoll() {
        let detector = VoiceEndpointDetector(configuration: configuration(
            confirmation: 0.02, preRoll: 0.02, trailing: 1, minimum: 0.02, maximum: 0.04))
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)

        let next = detector.process(frame(0.1, count: 20))
        guard let event = next.first,
              case let .speechStarted(_, audio) = event else {
            return XCTFail("expected speech to continue in a fresh utterance")
        }
        XCTAssertEqual(audio.samples.count, 20)
    }

    func testPreRollIsClampedToTrailingSilenceSoSpeechIsNeverReplayed() {
        // Carried-over pre-roll is taken from the tail of the utterance that
        // just ended, so a pre-roll longer than the trailing silence would feed
        // the previous instruction's speech back into the next recognizer.
        let config = configuration(confirmation: 0.02, preRoll: 0.05, trailing: 0.02, minimum: 0.02)
        XCTAssertEqual(config.preRollDuration, 0.02, accuracy: 0.0001)

        let detector = VoiceEndpointDetector(configuration: config)
        XCTAssertEqual(detector.process(frame(0.1, count: 20)).count, 1)
        XCTAssertEqual(detector.process(frame(0, count: 20)).count, 1)

        let next = detector.process(frame(0.1, count: 20))
        guard let event = next.first,
              case let .speechStarted(_, audio) = event else {
            return XCTFail("expected the next utterance to start")
        }
        // 20 silent pre-roll samples + the new 20, never the earlier speech.
        XCTAssertEqual(audio.samples.count, 40)
        XCTAssertEqual(Array(audio.samples.prefix(20)), Array(repeating: 0, count: 20))
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

    func testUtteranceQueuePreservesSpokenOrderAcrossOutOfOrderFinals() {
        var queue = VoiceUtteranceQueue()
        let first = UUID()
        let second = UUID()
        queue.begin(first)
        queue.begin(second)

        XCTAssertEqual(queue.resolve(second, as: .instruction("second")), [])
        XCTAssertEqual(queue.resolve(first, as: .instruction("first")), ["first", "second"])
        XCTAssertTrue(queue.isEmpty)
    }

    func testDiscardedUtteranceUnblocksLaterInstruction() {
        var queue = VoiceUtteranceQueue()
        let blip = UUID()
        let instruction = UUID()
        queue.begin(blip)
        queue.begin(instruction)

        XCTAssertEqual(queue.resolve(instruction, as: .instruction("continue")), [])
        XCTAssertEqual(queue.resolve(blip, as: .discarded), ["continue"])
        XCTAssertTrue(queue.isEmpty)
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
