# Krill Code Voice — implementation plan

**Depends on:** [KRILL_VOICE_VISION.md](KRILL_VOICE_VISION.md)
**Status:** Phase 3 implemented; ready for natural-turn-taking review
**Last updated:** 2026-08-01

## Architecture

Keep voice as an input/output layer around the existing agent harness. Do not
teach the coding model to consume raw audio when transcription is sufficient.

```text
NSPanel / SwiftUI surface
        │
        ▼
VoiceSessionCoordinator
  ├── CaptureEngine      (MicrophoneRecorder → continuous AVAudioEngine later)
  ├── EndpointDetector   (native energy VAD now → learned EOU later)
  ├── Transcriber        (Apple Speech / native MLX Whisper / Core ML Parakeet)
  ├── AgentController    (AgentLoop + AgentSession registry)
  ├── PermissionBridge   (native visual approval)
  └── SpeechOutput       (AVSpeech → Kokoro/PocketTTS)
        │
        ▼
Krill InferenceEngine (MLX/Metal) + KrillHarness
```

The coordinator owns a single explicit state machine:

```text
idle → listening → transcribing → working → speaking → idle
                      │              │          │
                      └── failed ◀───┴── interrupted
                                     │
                              awaitingApproval
```

Audio engine protocols must not import UI frameworks. The window observes
state/events on the main actor; model, tool, and transcription work runs in
tasks outside UI callbacks.

## Phase 0 — documentation and contract

Deliverables:

- Product parity mapping and native/local requirements.
- Prototype boundary and explicit deferrals.
- Requirement IDs used in code-review and release checklists.

Exit criteria:

- The vision distinguishes live Voice from dictation.
- The plan does not depend on a Python process or an unlicensed reference voice.

## Phase 1 — reviewable native prototype

### CLI entry

Add `--voice` to `krill code`. The existing command continues to resolve and
load the selected local model. Instead of entering the terminal renderer, it
hands the loaded `InferenceEngine`, tool registry, permission mode, and optional
initial task to a macOS voice-window controller.

Prototype behavior:

- `krill code <model> "<task>" --voice`
- A compact `NSPanel` floats above other apps and joins all Spaces.
- The first task may start immediately; subsequent instructions use the mic
  button or a text field.
- Closing the last panel terminates the prototype process cleanly.

### Reuse before invention

- Capture: `KrillCore/MicrophoneCapture.swift`
- STT: `KrillCore/SpeechRecognition.swift`
- TTS and spoken-text cleanup: `KrillCore/SpeechSynthesizer.swift`
- Agent runtime: `KrillHarness/AgentLoop.swift`
- Local generator: `KrillCLI/Code/EngineGenerator.swift`
- Tools: the same read/edit/bash/web registry as `CodeCommand`
- Permissions: implement a native `PermissionGate`; do not auto-approve an
  ask-mode action because it came from voice.

### Prototype UI

The first panel contains:

- Krill mark/title and active model.
- State pill and elapsed activity.
- Scrollable/selectable transcript.
- Text field for deterministic testing and accessibility.
- Mic/stop button, send button, interrupt button, and close control.
- Native approval sheet for gated tool calls.

Keep the prototype programmatic AppKit so it compiles in the existing SwiftPM
executable without introducing an Xcode-project dependency. A later SwiftUI app
target can reuse its coordinator.

### Prototype verification

- Unit-test pure state/status mapping and spoken event formatting.
- `swift build` succeeds on the supported macOS toolchain.
- Existing core/harness/CLI tests remain green.
- Manual smoke:
  1. Launch with a local model.
  2. Record “inspect this repository and tell me what it builds.”
  3. See the transcript.
  4. Observe read-only tool events.
  5. Hear and see the final response.
  6. In `ask` mode, request a file edit and verify the native approval blocks it.
  7. Interrupt a running task and close the panel.

## Phase 2 — decouple and harden voice engines

The TUI currently gates every voice posture on `engine.canUseNativeAudio`.
Change the capability rules:

- `dictate` and `handsfree` require an STT engine, not a multimodal LLM.
- `send` alone requires model audio input.
- TTS is independent of both.

Extract protocols:

```swift
protocol VoiceTranscriber: Sendable {
    func transcribe(_ wav: Data) async throws -> String
}

protocol VoiceActivityDetector: Sendable {
    func accept(_ samples: UnsafeBufferPointer<Float>) -> VoiceActivityEvent
}

protocol VoiceOutput: Sendable {
    func speak(_ text: String) async throws
    func stop()
}
```

Add configuration for engine, language, voice identifier, rate, downloaded
model SKU, and automatic narration policy. Keep Apple engines as fallbacks.

### Phase 2 implementation note

Implemented on 2026-08-01:

- `VoiceTranscriber`, `VoiceActivityDetector`, and `VoiceOutput` now define
  UI-independent native voice boundaries in `KrillCore`.
- `dictate` and `handsfree` work with text-only coding models through Apple
  Speech or Krill's native MLX Whisper runtime. Only `send` and raw audio
  attachments require an audio-capable model.
- Apple recognition accepts a configured locale. Apple speech output accepts a
  voice identifier and rate, with safe language/system fallbacks.
- TOML and `KRILL_*` configuration cover the engine, language, voice,
  narration policy, and Whisper model SKU. The TUI applies all of them when a
  session starts; the Phase 1 panel applies its Apple language/voice/rate and
  narration settings while selectable panel STT remains follow-up work.
- At the Phase 2 boundary the VAD protocol was deliberately unimplemented;
  Phase 3 supplies the first native detector and continuous capture consumer.

## Phase 3 — natural turn-taking

Recommended native stack:

- Streaming STT/end-of-utterance: FluidAudio Parakeet EOU on Core ML/ANE.
- VAD fallback: Silero VAD through Core ML.
- Coding model: existing MLX Swift engine on Metal.
- TTS: PocketTTS for streaming/custom voice, Kokoro for fast preset voices.

Work items:

- Replace click endpointing with continuous capture and EOU/VAD.
- Render partial transcripts without committing them to agent history.
- Stop audio output immediately on detected user speech.
- Enable AVAudioEngine voice processing/echo cancellation where supported.
- Chunk final speech at semantic sentence boundaries.
- Add a narration prompt/policy that summarizes tool-heavy results without
  losing the full visual transcript.

### Phase 3 implementation note

Implemented on 2026-08-02 for the floating `krill code --voice` surface:

- A separate continuous `AVAudioEngine` capture path feeds a serial native
  energy detector with confirmation, hysteresis, pre-roll, minimum speech,
  trailing-silence endpointing, and a maximum utterance duration.
- Apple on-device streaming recognition publishes provisional partial text to
  the panel. Only the final endpointed transcript is committed to agent history;
  the existing one-shot recognizer is the final fallback.
- Capture remains active while the foreground agent works. Completed spoken
  steering is displayed and queued in order, then submitted at the next safe
  `AgentLoop` boundary. The current loop does not support unsafe mid-generation
  prompt injection.
- Confirmed user speech stops queued TTS immediately. The capture engine attempts
  macOS voice processing and the panel recommends headphones when the active
  route cannot provide it.
- Final narration is split at semantic sentence boundaries and queued through
  one local Apple synthesizer request, preserving interruption between replies.
- Approval requests are scoped to their agent run and remain visual-only; speech
  can queue an instruction but cannot approve a tool.

This is a native baseline, not the final acoustic model. FluidAudio/Parakeet EOU
or a Core ML VAD can replace the energy detector behind the same boundary after
on-device quality and resource measurements. The prototype also copies each
microphone buffer before leaving AVAudioEngine's callback; a preallocated
single-producer/single-consumer ring remains a productization requirement for
strict real-time audio behavior.

Latency budgets for a warm session:

- State reaction/UI feedback: under 100 ms.
- Partial transcript update: under 300 ms.
- End-of-utterance commit: 300–700 ms after speech ends.
- First spoken acknowledgement: target under 1.5 s when no model tool call is
  required; longer work acknowledges immediately and continues asynchronously.

## Phase 4 — real multi-agent voice coordination

Implement the related Review Desk follow-ups as shared infrastructure:

1. Rich session state: `working`, `blocked`, `done`, `failed`, `idle`.
2. Stable session ID/title plus event stream.
3. An in-process controller and authenticated local Unix socket for list, start,
   steer, approve, cancel, wait, and collect.
4. Awaitable `dispatch_agent` so a child can return a bounded result to its
   parent.
5. Queue steering messages at safe agent-loop boundaries.
6. Add voice intents for “ask the test agent…”, “cancel agent two”, and “what is
   blocked?” without hiding the underlying text instruction.

The floating panel becomes a client of that controller. This prevents agent
logic from being duplicated in AppKit and later allows a menu-bar app, CLI, or
remote client to share the same sessions.

## Phase 5 — productization

- Dedicated signed `.app` with microphone, speech-recognition, accessibility,
  and optional global-hotkey usage strings.
- Menu-bar presence and configurable global hotkey.
- Download manager with hashes, licenses, size, progress, and removal.
- On-device custom-voice enrollment with explicit consent confirmation.
- Crash recovery and resumable session state.
- Screen context only after explicit per-capture consent and a visible capture
  indicator.
- Accessibility audit, VoiceOver labels, keyboard-only operation, localization,
  and reduced-motion support.
- Performance tests on supported Apple Silicon memory tiers.

## Test strategy

### Unit

- Voice state transitions and illegal-transition handling.
- Transcript/event reduction.
- Spoken-summary cleanup and maximum narration length.
- Permission requests never resolve without explicit user action.
- Engine capability routing (`dictate` versus raw-audio `send`).

### Integration

- Deterministic fake transcriber → fake agent → fake TTS session.
- Real `AgentLoop` with mock generator and tools.
- Cancellation during capture, transcription, generation, tool approval, and
  speech.
- Two background sessions with serialized generation and independent steering.

### Live/on-device

- Quiet/noisy microphones, headphones, and speaker echo.
- Short utterances, pauses, self-corrections, and barge-in.
- Apple system STT/TTS fallback with no downloaded models.
- Each downloadable engine on the minimum supported Mac.
- Ten-minute coding-session dogfood scenario from the vision success criteria.

## Prototype deferrals and review questions

The current prototype intentionally defers global hotkeys, screen context,
downloadable neural voices, subagent cards, true mid-run agent injection, and
restart recovery.

The review should decide:

1. Is a compact transcript panel the right shape, or should the default be a
   smaller voice orb that expands only on demand?
2. Should Krill acknowledge every instruction aloud, or remain silent until a
   blocker/result exists?
3. Is the default posture `ask` or `accept-edits` for a voice-launched coding
   session?
4. Should a custom voice pack be a first-class v1 feature, or follow the natural
   preset voice?
