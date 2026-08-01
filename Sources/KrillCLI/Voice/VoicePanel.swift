import AppKit
import Foundation
import KrillCore
import KrillHarness
import KrillRegistry

/// Native, always-available voice surface for `krill code --voice`. The panel
/// owns presentation and turn scheduling only; capture, endpointing, live STT,
/// TTS, and the agent loop remain independently replaceable native seams.
@MainActor
enum VoicePanel {
    // AppKit does not retain a window controller for us through the nested run
    // loop. Keep this strong reference until the panel is explicitly closed.
    private static var activeController: VoicePanelController?

    static func run(
        loop: AgentLoop,
        initialTask: String,
        system: String?,
        modelName: String,
        permissionMode: PermissionMode,
        voiceLanguage: String = "auto",
        voiceIdentifier: String = "",
        voiceRate: Float = AppleSpeechSettings.systemRate,
        narration: VoiceNarrationPolicy = .final
    ) {
        let application = NSApplication.shared
        application.setActivationPolicy(.accessory)

        let controller = VoicePanelController(
            loop: loop, system: system, modelName: modelName,
            permissionMode: permissionMode, voiceLanguage: voiceLanguage,
            voiceIdentifier: voiceIdentifier, voiceRate: voiceRate,
            narration: narration)
        activeController = controller
        controller.show(initialTask: initialTask)
        application.activate(ignoringOtherApps: true)
        application.run()
        activeController = nil
    }
}

/// A thread-safe event handoff. `AgentLoop` calls its callback synchronously in
/// event order, while AppKit rendering must happen on the main actor. Draining
/// the complete buffer before clearing a run keeps the terminal event visible.
private final class VoiceAgentEventBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [AgentEvent] = []

    func push(_ event: AgentEvent) {
        lock.lock(); events.append(event); lock.unlock()
    }

    func drain() -> [AgentEvent] {
        lock.lock(); defer { lock.unlock() }
        let pending = events
        events.removeAll(keepingCapacity: true)
        return pending
    }
}

/// AppKit controller kept on the main actor. The microphone callback arrives
/// off-main, then explicitly hops here before it may affect UI or turn state.
@MainActor
private final class VoicePanelController: NSObject, NSWindowDelegate, NSTextFieldDelegate {
    private enum UtteranceResolution {
        case instruction(String)
        case discarded
    }

    private let loop: AgentLoop
    private let system: String?
    private let modelName: String
    private let permissionMode: PermissionMode
    private let speechSettings: AppleSpeechSettings
    private let recognizer: SpeechRecognizer
    private let speaker: SpeechSynthesizer
    private let narration: VoiceNarrationPolicy
    private let endpointDetector = VoiceEndpointDetector()

    private lazy var microphone = ContinuousMicrophoneCapture { _ in }

    private var transcriptMessages: [[String: String]] = []
    private var queuedInstructions: [String] = []
    private var workTask: Task<Void, Never>?
    private var narrationTask: Task<Void, Never>?
    private var permissionTask: Task<Void, Never>?
    private var activeRunID: UUID?
    private var eventBuffer: VoiceAgentEventBuffer?
    private var agentChipShown = false
    private var pendingApproval: (runID: UUID, continuation: CheckedContinuation<Bool, Never>)?

    // Each endpointed clip retains its own recognizer and finalisation task so
    // a quick follow-up sentence cannot overwrite a still-arriving final result.
    private var activeSpeechID: UUID?
    private var liveRecognizers: [UUID: LiveAppleSpeechRecognizer] = [:]
    private var endpointedAudio: [UUID: VoiceAudioFrame] = [:]
    private var liveFinals: [UUID: String] = [:]
    private var finalizationTasks: [UUID: Task<Void, Never>] = [:]
    private var utteranceOrder: [UUID] = []
    private var utteranceResolutions: [UUID: UtteranceResolution] = [:]
    private var provisionalTranscript = ""
    private var provisionalSpeechID: UUID?

    private var isListening = false
    private var permissionsGranted: Bool?
    private var captureGeneration: UInt64 = 0
    private var closing = false
    private var lastLevel: VoiceSignalLevel?
    private var lastError: String?

    private var panel: NSPanel!
    private var transcriptView: NSTextView!
    private var inputField: NSTextField!
    private var statusLabel: NSTextField!
    private var partialLabel: NSTextField!
    private var aecLabel: NSTextField!
    private var recordButton: NSButton!
    private var sendButton: NSButton!
    private var interruptButton: NSButton!
    private var approvalBox: NSStackView!
    private var approvalLabel: NSTextField!

    init(loop: AgentLoop, system: String?, modelName: String, permissionMode: PermissionMode,
         voiceLanguage: String, voiceIdentifier: String, voiceRate: Float,
         narration: VoiceNarrationPolicy) {
        self.loop = loop
        self.system = system
        self.modelName = modelName
        self.permissionMode = permissionMode
        self.speechSettings = AppleSpeechSettings(
            language: voiceLanguage, voiceIdentifier: voiceIdentifier, rate: voiceRate)
        self.recognizer = SpeechRecognizer(language: voiceLanguage)
        self.speaker = SpeechSynthesizer(
            language: voiceLanguage, voiceIdentifier: voiceIdentifier, rate: voiceRate)
        self.narration = narration
        super.init()
    }

    func show(initialTask: String) {
        buildPanel()
        panel.makeKeyAndOrderFront(nil)
        append(.note("Voice Code ready — \(permissionMode.postureNote)"))
        beginListening()
        if !initialTask.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            acceptInstruction(initialTask, source: "initial")
        }
    }

    private func buildPanel() {
        let rect = NSRect(x: 0, y: 0, width: 680, height: 660)
        panel = NSPanel(
            contentRect: rect,
            styleMask: [.titled, .closable, .resizable, .utilityWindow],
            backing: .buffered, defer: false)
        panel.title = "Krill Code Voice — \(modelName)"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.delegate = self
        panel.center()

        let content = NSView()
        content.translatesAutoresizingMaskIntoConstraints = false
        panel.contentView = content

        statusLabel = NSTextField(labelWithString: "Preparing voice…")
        statusLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        statusLabel.textColor = .secondaryLabelColor

        partialLabel = NSTextField(wrappingLabelWithString: "")
        partialLabel.font = .systemFont(ofSize: 12)
        partialLabel.textColor = .tertiaryLabelColor
        partialLabel.maximumNumberOfLines = 2
        partialLabel.isHidden = true

        aecLabel = NSTextField(wrappingLabelWithString: "")
        aecLabel.font = .systemFont(ofSize: 11)
        aecLabel.textColor = .systemOrange
        aecLabel.isHidden = true

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        transcriptView = NSTextView()
        transcriptView.isEditable = false
        transcriptView.isSelectable = true
        transcriptView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        transcriptView.textContainerInset = NSSize(width: 10, height: 10)
        transcriptView.autoresizingMask = [.width]
        scroll.documentView = transcriptView

        approvalLabel = NSTextField(wrappingLabelWithString: "")
        approvalLabel.font = .systemFont(ofSize: 12)
        let allow = NSButton(title: "Allow", target: self, action: #selector(allowApproval))
        let deny = NSButton(title: "Deny", target: self, action: #selector(denyApproval))
        approvalBox = NSStackView(views: [approvalLabel, allow, deny])
        approvalBox.orientation = .horizontal
        approvalBox.alignment = .centerY
        approvalBox.spacing = 8
        approvalBox.isHidden = true

        inputField = NSTextField()
        inputField.placeholderString = "Type an instruction, even while Krill is working…"
        inputField.delegate = self
        inputField.target = self
        inputField.action = #selector(sendTypedTurn)

        recordButton = NSButton(title: "Pause conversation", target: self, action: #selector(toggleListening))
        sendButton = NSButton(title: "Send", target: self, action: #selector(sendTypedTurn))
        interruptButton = NSButton(title: "Interrupt", target: self, action: #selector(interrupt))
        interruptButton.isEnabled = false
        let controls = NSStackView(views: [recordButton, inputField, sendButton, interruptButton])
        controls.orientation = .horizontal
        controls.alignment = .centerY
        controls.spacing = 8
        inputField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        inputField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let stack = NSStackView(views: [statusLabel, partialLabel, aecLabel, scroll, approvalBox, controls])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 14),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -14),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 14),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -14),
            scroll.widthAnchor.constraint(equalTo: stack.widthAnchor),
            scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 360),
            partialLabel.widthAnchor.constraint(equalTo: stack.widthAnchor),
            aecLabel.widthAnchor.constraint(equalTo: stack.widthAnchor),
            approvalBox.widthAnchor.constraint(equalTo: stack.widthAnchor),
            controls.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])
        refreshUI()
    }

    func controlTextDidEndEditing(_ obj: Notification) {
        if obj.object as? NSTextField === inputField { sendTypedTurn() }
    }

    @objc private func sendTypedTurn() {
        let text = inputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        inputField.stringValue = ""
        guard !text.isEmpty else { return }
        speaker.stop()
        acceptInstruction(text, source: "typed")
    }

    /// The sole voice control pauses/resumes microphone/STT conversation. It
    /// never cancels an agent run; the explicit Interrupt button owns that.
    @objc private func toggleListening() {
        if isListening { pauseListening() } else { beginListening() }
    }

    private func beginListening() {
        guard !closing, !isListening, permissionTask == nil else { return }
        if permissionsGranted == true {
            startCapture()
            return
        }
        guard permissionsGranted == nil else {
            fail("Microphone and Speech Recognition access are required to resume voice conversation.")
            return
        }
        recordButton.isEnabled = false
        recordButton.title = "Preparing voice…"
        permissionTask = Task { [weak self] in
            guard let self else { return }
            let microphoneAllowed = await MicrophoneRecorder.requestAccess()
            let speechAllowed: Bool
            if microphoneAllowed {
                speechAllowed = await SpeechRecognizer.requestAuthorization()
            } else {
                speechAllowed = false
            }
            guard !Task.isCancelled, !self.closing else { return }
            self.permissionTask = nil
            self.permissionsGranted = microphoneAllowed && speechAllowed
            guard self.permissionsGranted == true else {
                self.fail("Voice access was denied. Enable Microphone and Speech Recognition in System Settings, then reopen this panel.")
                return
            }
            self.startCapture()
        }
    }

    private func startCapture() {
        guard !closing, !isListening else { return }
        do {
            captureGeneration &+= 1
            let generation = captureGeneration
            microphone.setAudioFrameHandler { [weak self] frame in
                Task { @MainActor [weak self] in
                    self?.receiveAudioFrame(frame, captureGeneration: generation)
                }
            }
            try microphone.start()
            isListening = true
            lastError = nil
            if microphone.isVoiceProcessingEnabled {
                aecLabel.isHidden = true
            } else {
                aecLabel.stringValue = "Echo processing is unavailable on this input — headphones are recommended."
                aecLabel.isHidden = false
            }
            refreshUI()
        } catch {
            fail(error.localizedDescription)
        }
    }

    private func pauseListening() {
        // Invalidate both frames waiting on the capture processing queue and
        // MainActor tasks already created by the previous listening session.
        captureGeneration &+= 1
        microphone.stop()
        endpointDetector.reset()
        isListening = false
        activeSpeechID = nil
        provisionalTranscript = ""
        provisionalSpeechID = nil
        partialLabel.isHidden = true
        for recognizer in liveRecognizers.values { recognizer.cancel() }
        liveRecognizers.removeAll()
        endpointedAudio.removeAll()
        liveFinals.removeAll()
        utteranceOrder.removeAll()
        utteranceResolutions.removeAll()
        for task in finalizationTasks.values { task.cancel() }
        finalizationTasks.removeAll()
        speaker.stop()
        refreshUI()
    }

    private func receiveAudioFrame(_ frame: VoiceAudioFrame, captureGeneration: UInt64) {
        guard isListening, !closing, captureGeneration == self.captureGeneration else { return }
        for event in endpointDetector.process(frame) {
            switch event {
            case let .speechStarted(level, audio):
                beginUtterance(level: level, audio: audio)
            case let .speechContinued(level, audio):
                lastLevel = level
                if let id = activeSpeechID { liveRecognizers[id]?.append(audio) }
                refreshUI()
            case let .utteranceReady(samples, sampleRate):
                finishUtterance(samples: samples, sampleRate: sampleRate)
            }
        }
    }

    private func beginUtterance(level: VoiceSignalLevel, audio: VoiceAudioFrame) {
        // Confirmed VAD speech (not a raw audio frame) is the barge-in trigger.
        // It cannot reach the approval gate and it stops only local narration.
        speaker.stop()
        narrationTask?.cancel()
        lastLevel = level
        let id = UUID()
        activeSpeechID = id
        utteranceOrder.append(id)
        provisionalTranscript = ""
        provisionalSpeechID = id
        refreshPartialTranscript()
        let live = LiveAppleSpeechRecognizer(settings: speechSettings) { [weak self] update in
            Task { @MainActor [weak self] in self?.receiveLiveUpdate(update, utteranceID: id) }
        }
        liveRecognizers[id] = live
        do {
            try live.start()
            live.append(audio)
        } catch {
            // Final one-shot transcription remains available when live Apple
            // Speech is temporarily unavailable; the error is non-terminal.
            append(.note("Live transcript unavailable; confirming this utterance after it ends."))
        }
        refreshUI()
    }

    private func receiveLiveUpdate(_ update: LiveSpeechTranscriptionUpdate, utteranceID: UUID) {
        guard !closing, liveRecognizers[utteranceID] != nil else { return }
        if update.isFinal {
            liveFinals[utteranceID] = update.text
            if endpointedAudio[utteranceID] != nil { commitUtterance(utteranceID, text: update.text) }
            return
        }
        guard activeSpeechID == utteranceID else { return }
        provisionalTranscript = update.text
        refreshPartialTranscript()
    }

    private func finishUtterance(samples: [Float], sampleRate: Double) {
        guard let id = activeSpeechID else { return }
        activeSpeechID = nil
        let audio = VoiceAudioFrame(samples: samples, sampleRate: sampleRate)
        endpointedAudio[id] = audio
        liveRecognizers[id]?.finish()
        if let final = liveFinals[id] {
            commitUtterance(id, text: final)
            return
        }
        // Apple may publish a final just after endAudio. Give it a bounded
        // window, then use the existing local WAV recognizer as a safe fallback.
        finalizationTasks[id] = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(900))
            guard !Task.isCancelled else { return }
            await self?.fallbackOrCommitUtterance(id)
        }
        refreshUI()
    }

    private func fallbackOrCommitUtterance(_ id: UUID) async {
        guard !closing, let audio = endpointedAudio[id] else { return }
        if let final = liveFinals[id] {
            commitUtterance(id, text: final)
            return
        }
        let wav = MediaAttachment.encodeWAV(
            samples: audio.samples, sampleRate: max(1, Int(audio.sampleRate.rounded())))
        let fallback = await recognizer.transcribe(wav: wav)
        // Pausing/closing removes the endpoint record and cancellation can race
        // the non-cancellable Speech callback. Those paths are intentional and
        // must not surface as recognition failures.
        guard !Task.isCancelled, !closing, endpointedAudio[id] != nil else { return }
        guard let fallback else {
            resolveUtterance(id, as: .discarded)
            append(.note("I heard speech but could not confirm a local transcription."))
            return
        }
        commitUtterance(id, text: fallback)
    }

    /// Only an endpointed final (or its one-shot fallback) reaches the agent.
    /// Partial captions are strictly presentation state and never enter history.
    private func commitUtterance(_ id: UUID, text: String) {
        guard endpointedAudio[id] != nil else { return }
        let confirmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !confirmed.isEmpty else {
            resolveUtterance(id, as: .discarded)
            return
        }
        resolveUtterance(id, as: .instruction(confirmed))
    }

    private func resolveUtterance(_ id: UUID, as resolution: UtteranceResolution) {
        guard utteranceOrder.contains(id), utteranceResolutions[id] == nil else { return }
        cleanupUtterance(id)
        utteranceResolutions[id] = resolution
        drainResolvedUtterances()
    }

    /// Preserve spoken order even when Apple's live final for a later utterance
    /// beats an earlier turn's one-shot fallback.
    private func drainResolvedUtterances() {
        while let first = utteranceOrder.first,
              let resolution = utteranceResolutions.removeValue(forKey: first) {
            utteranceOrder.removeFirst()
            if case let .instruction(text) = resolution {
                acceptInstruction(text, source: "spoken")
            }
        }
    }

    private func cleanupUtterance(_ id: UUID) {
        finalizationTasks.removeValue(forKey: id)?.cancel()
        liveRecognizers.removeValue(forKey: id)?.cancel()
        endpointedAudio.removeValue(forKey: id)
        liveFinals.removeValue(forKey: id)
        if activeSpeechID == id { activeSpeechID = nil }
        // A previous utterance may finish its one-shot fallback after the user
        // has already started speaking again. Never let that stale cleanup erase
        // the newer utterance's provisional caption.
        if provisionalSpeechID == id {
            provisionalSpeechID = nil
            provisionalTranscript = ""
            refreshPartialTranscript()
        }
        refreshUI()
    }

    private func refreshPartialTranscript() {
        let text = provisionalTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        partialLabel.stringValue = text.isEmpty ? "" : "Hearing (not sent): \(text)"
        partialLabel.isHidden = text.isEmpty
    }

    private func acceptInstruction(_ text: String, source: String) {
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !closing else { return }
        if workTask != nil {
            queuedInstructions.append(prompt)
            append(.note("Queued \(source) instruction #\(queuedInstructions.count): \(prompt)"))
            refreshUI()
            return
        }
        startRun(prompt)
    }

    private func startRun(_ prompt: String) {
        guard !closing, workTask == nil else {
            queuedInstructions.append(prompt)
            refreshUI()
            return
        }
        speaker.stop()
        narrationTask?.cancel()
        append(.user(prompt))
        let runID = UUID()
        let buffer = VoiceAgentEventBuffer()
        activeRunID = runID
        eventBuffer = buffer
        agentChipShown = false
        let gate = VoicePermissionGate { [weak self] toolName, argumentsJSON in
            guard let self else { return false }
            return await self.requestApproval(
                runID: runID, toolName: toolName, argumentsJSON: argumentsJSON)
        }
        let turnLoop = AgentLoop(
            generator: loop.generator, tools: loop.tools,
            maxIterations: loop.maxIterations, constrainToolArgs: loop.constrainToolArgs,
            permission: loop.permission, gate: gate)
        workTask = Task { [weak self] in
            guard let self else { return }
            let transcript = await turnLoop.run(
                user: prompt, system: self.system, priorMessages: self.transcriptMessages,
                onEvent: { event in
                    buffer.push(event)
                    Task { @MainActor [weak self] in self?.drainAgentEvents(runID: runID) }
                })
            self.finishRun(transcript, runID: runID)
        }
        refreshUI()
    }

    private func drainAgentEvents(runID: UUID) {
        guard activeRunID == runID, let buffer = eventBuffer else { return }
        for event in buffer.drain() { render(event, runID: runID) }
    }

    private func render(_ event: AgentEvent, runID: UUID) {
        guard activeRunID == runID else { return }
        var entries: [AgentEntry] = []
        foldAgentEvent(event, into: &entries, chipShown: &agentChipShown)
        for entry in entries { append(entry) }
        refreshUI()
    }

    private func finishRun(_ transcript: AgentTranscript, runID: UUID) {
        // This is deliberately before clearing activeRunID: queued main-actor
        // rendering tasks may still exist, but every event already emitted by
        // AgentLoop is synchronously present in this buffer.
        drainAgentEvents(runID: runID)
        guard activeRunID == runID else { return }
        resolveApproval(false, expectedRunID: runID)
        transcriptMessages = transcript.messages
        workTask = nil
        activeRunID = nil
        eventBuffer = nil

        let hasQueuedSteering = !queuedInstructions.isEmpty
        if !transcript.wasCancelled, transcript.hitIterationLimit {
            append(.note("Stopped at the iteration limit."))
        }
        if hasQueuedSteering {
            // Do not narrate a now-superseded answer. The queued instruction is
            // submitted at this completed-run boundary, never mid-tool-call.
            speaker.stop()
            startNextQueuedTurn()
            return
        }
        if !transcript.wasCancelled, !transcript.finalText.isEmpty {
            speakFinal(transcript.finalText)
        }
        refreshUI()
    }

    private func startNextQueuedTurn() {
        guard workTask == nil, !queuedInstructions.isEmpty, !closing else { return }
        let next = queuedInstructions.removeFirst()
        startRun(next)
    }

    private func speakFinal(_ text: String) {
        guard narration != .off, SpeechSynthesizer.isAvailable, !text.isEmpty else {
            refreshUI()
            return
        }
        narrationTask?.cancel()
        var chunker = SpeechSentenceChunker()
        let chunks = chunker.append(text) + chunker.finish()
        // Sentence chunks are enqueued as one reply. VAD-confirmed speech calls
        // `stop()` immediately before a future chunk may begin (barge-in).
        speaker.speakChunks(chunks)
        narrationTask = Task { [weak self] in
            guard let self else { return }
            while self.speaker.isSpeaking {
                try? await Task.sleep(for: .milliseconds(180))
                if Task.isCancelled { return }
            }
            self.refreshUI()
        }
        refreshUI()
    }

    private func requestApproval(runID: UUID, toolName: String, argumentsJSON: String) async -> Bool {
        guard !closing, !Task.isCancelled, activeRunID == runID, workTask != nil,
              pendingApproval == nil else { return false }
        approvalLabel.stringValue = "Allow \(toolName)(\(argumentsJSON))?"
        approvalBox.isHidden = false
        refreshUI()
        return await withCheckedContinuation { continuation in
            // The gate can be reached after a cancellation race. Recheck in the
            // continuation creation path and fail closed for that stale run.
            guard !closing, activeRunID == runID, workTask != nil else {
                continuation.resume(returning: false)
                return
            }
            pendingApproval = (runID, continuation)
        }
    }

    @objc private func allowApproval() { resolveApproval(true) }
    @objc private func denyApproval() { resolveApproval(false) }

    private func resolveApproval(_ allowed: Bool, expectedRunID: UUID? = nil) {
        guard let pendingApproval else { return }
        if let expectedRunID, pendingApproval.runID != expectedRunID { return }
        self.pendingApproval = nil
        approvalBox.isHidden = true
        // Stale/cancelled runs can never be approved, even if an old button
        // action arrives after a new run has already begun.
        let valid = !closing && activeRunID == pendingApproval.runID && workTask != nil
        pendingApproval.continuation.resume(returning: allowed && valid)
        refreshUI()
    }

    @objc private func interrupt() {
        // Interrupt cancels agent work and narration only. Continuous capture
        // remains active, so a following spoken instruction is naturally queued
        // until the cancelled AgentLoop returns at its safe boundary.
        speaker.stop()
        narrationTask?.cancel()
        if let runID = activeRunID {
            resolveApproval(false, expectedRunID: runID)
            workTask?.cancel()
            append(.note("Interrupt requested; finishing cancellation before the next queued instruction."))
        }
        refreshUI()
    }

    func windowWillClose(_ notification: Notification) {
        closing = true
        permissionTask?.cancel()
        pauseListening()
        speaker.stop()
        narrationTask?.cancel()
        if let runID = activeRunID { resolveApproval(false, expectedRunID: runID) }
        workTask?.cancel()
        NSApplication.shared.stop(nil)
    }

    private func refreshUI() {
        guard panel != nil else { return }
        var states: [String] = [isListening ? "Listening" : "Paused"]
        if activeSpeechID != nil { states.append("speech detected") }
        if pendingApproval != nil { states.append("awaiting approval") }
        else if workTask != nil { states.append("working") }
        if speaker.isSpeaking { states.append("speaking") }
        if !queuedInstructions.isEmpty { states.append("\(queuedInstructions.count) queued") }
        if let level = lastLevel, isListening { states.append(String(format: "%.0f dB", level.decibels)) }
        statusLabel.stringValue = "\(states.joined(separator: " · ")) · \(permissionMode.label)"
        statusLabel.textColor = lastError == nil ? .secondaryLabelColor : .systemRed
        recordButton.title = isListening ? "Pause conversation" : "Resume conversation"
        recordButton.isEnabled = !closing && permissionTask == nil
        // Typed follow-ups intentionally stay enabled while an agent is running;
        // they are queued just like completed spoken instructions.
        inputField.isEnabled = !closing
        sendButton.isEnabled = !closing
        interruptButton.isEnabled = workTask != nil || pendingApproval != nil || speaker.isSpeaking
    }

    private func fail(_ message: String) {
        lastError = message
        append(.note("Error: \(message)"))
        refreshUI()
    }

    private func append(_ entry: AgentEntry) {
        let line: String
        switch entry {
        case .user(let text): line = "You\n\(text)"
        case .assistant(let text): line = "Krill\n\(text)"
        case .toolCall(let name, let args): line = "Tool → \(name)(\(args))"
        case .toolResult(let content, let isError):
            line = "\(isError ? "Tool error" : "Tool result")\n\(content)"
        case .note(let text): line = text
        }
        transcriptView.textStorage?.append(NSAttributedString(string: line + "\n\n"))
        transcriptView.scrollToEndOfDocument(nil)
    }
}

/// Bridges the harness's sendable approval seam to the main-actor AppKit panel.
/// It does not receive transcript text, so spoken words cannot approve a tool.
private final class VoicePermissionGate: PermissionGate, @unchecked Sendable {
    private let request: @Sendable (String, String) async -> Bool

    init(request: @escaping @Sendable (String, String) async -> Bool) {
        self.request = request
    }

    func approve(toolName: String, argumentsJSON: String) async -> Bool {
        await request(toolName, argumentsJSON)
    }
}
