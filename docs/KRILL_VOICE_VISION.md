# Krill Code Voice — product vision and requirements

**Status:** proposed; Phase 3 natural-turn-taking prototype ready for review
**Owner:** Krill
**Last reviewed:** 2026-08-01

## Product statement

Krill Code Voice is a small, always-available macOS conversation surface for
driving `krill code` without returning to the terminal. The user talks normally,
Krill turns speech into coding instructions, the main agent can delegate bounded
work to background agents, and the voice session remains available to explain,
interrupt, redirect, approve, or cancel that work.

The experience is local-first and Mac-native. Speech capture, endpointing,
transcription, agent orchestration, speech synthesis, and the floating interface
must have a native Swift, Core ML, or MLX path. A Python sidecar is not part of
the shipping architecture.

## Reference feature released by OpenAI

The parity target is the recently released **ChatGPT Voice for Chat, Work, and
Codex in the ChatGPT desktop app**, not simple voice dictation. The official
[ChatGPT Voice documentation](https://learn.chatgpt.com/docs/features/voice.md)
describes the behaviors Krill is targeting:

- Begin a new chat or task in voice mode.
- Use natural turn-taking and interrupt a spoken response.
- Continue talking after work starts to check progress or change direction.
- Start separate threads for longer tasks, inspect existing threads, and send
  follow-up instructions.
- Bring progress, blockers, and results back into the voice conversation.
- Apply the same task permissions whether instructions arrive through text or
  voice.
- Start voice from a configurable keyboard shortcut.
- Optionally use visible screen context on macOS.

The desktop app also supports popping a chat into a separate window and keeping
it always on top. Krill combines that window behavior with the Codex Voice
coordination behavior into one compact coding surface.

This is a behavioral reference, not an API dependency. Krill will not copy
OpenAI branding, voices, or proprietary UI assets.

## Desired experience

1. The user runs `krill code <model> --voice` or presses a global hotkey.
2. A compact floating panel appears above the current editor or terminal.
3. The user speaks naturally. Partial transcription is visible immediately.
4. A completed utterance becomes a normal Krill agent instruction.
5. Krill narrates only useful information: its acknowledgement, material
   blockers, approval requests, and a concise final result.
6. The main agent can launch focused background agents. The panel shows each
   agent as `working`, `blocked`, `done`, `failed`, or `idle`.
7. The user can interrupt speech, steer the main task, or address a particular
   agent while other work continues.
8. Every important action remains visible in text. Voice is an additional
   control surface, never the sole audit trail.

The tone should feel like speaking with a calm collaborator. Krill must not read
raw tool calls, code blocks, token counters, or long terminal output aloud.

## Existing foundation

Krill already provides much of the engine beneath this experience:

- The unified chat/agent TUI, `/agent`, and `krill code` are documented in
  [TUI.md](TUI.md).
- Push-to-talk `dictate`, `handsfree`, and `send` modes already use Apple Speech
  or Krill's native MLX Whisper runtime.
- `SpeechSynthesizer` already reads replies using local AVFoundation voices.
- `dispatch_agent` and `AgentSession` already create switchable background
  workers.
- The Review Desk plan already calls for a richer agent-state model, attention
  signals, a control socket, and awaitable dispatch; see
  [REVIEW_DESK_PLAN.md](REVIEW_DESK_PLAN.md#herdr-extractions-decided-no-multiplexer--steal-the-ideas-not-the-panes).

Phase 2 separates local transcription from model audio capability: Apple Speech
and native MLX Whisper dictation work with text-only coding models, while
raw-audio `send` remains limited to audio-capable models. Phase 3 adds continuous
capture, native endpointing, streaming Apple partials, safe-boundary steering,
barge-in, and chunked narration to the floating panel. The terminal TUI remains
push-to-talk; true in-flight agent injection and background-agent control remain
later coordination work.

## Functional requirements

### Voice session

- **VOICE-001 — Dedicated entry point.** `krill code` can launch a voice session
  directly without first navigating TUI commands.
- **VOICE-002 — Floating panel.** The session uses a compact native macOS panel
  that can remain above other applications and appear on every Space.
- **VOICE-003 — Visible state.** The panel always distinguishes `idle`,
  `listening`, `transcribing`, `working`, `awaiting approval`, `speaking`,
  `interrupted`, and `failed`.
- **VOICE-004 — Transcript.** User speech, agent summaries, approvals, and errors
  remain visible and selectable.
- **VOICE-005 — Hotkey.** A configurable shortcut opens the panel and begins or
  stops listening. The prototype may use an in-panel button before global-hotkey
  permissions are added.

### Natural conversation

- **CONV-001 — Endpointing.** The production system detects the end of an
  utterance without requiring a key release.
- **CONV-002 — Barge-in.** New user speech immediately stops TTS playback and is
  treated as a new steering instruction.
- **CONV-003 — Streaming feedback.** Partial transcription appears while the
  user speaks; synthesized speech starts from safe sentence chunks instead of
  waiting for an entire long answer.
- **CONV-004 — Echo resistance.** Speaker output must not be re-ingested as user
  speech. Use macOS voice processing/echo cancellation where available and
  pause or gate recognition during unsafe playback states.
- **CONV-005 — Concise narration.** Coding output is summarized for speech while
  the complete answer remains on screen.

### Coding-agent coordination

- **AGENT-001 — Normal agent semantics.** A voice instruction enters the same
  `AgentLoop`, tool registry, working directory, model history, and permission
  policy as a typed instruction.
- **AGENT-002 — Continue while working.** The voice session remains responsive
  while an agent is running and can queue or inject steering messages.
- **AGENT-003 — Delegate.** The main agent can launch multiple focused agents and
  the user can explicitly request delegation by voice.
- **AGENT-004 — Addressable agents.** Each worker has a stable ID/title and can
  receive a follow-up, cancellation, or approval.
- **AGENT-005 — Collect results.** A delegated worker can return a bounded result
  to its parent. Fire-and-forget remains available but is not the only mode.
- **AGENT-006 — Status narration.** Transitions to blocked, failed, and done are
  surfaced visually; material transitions may be spoken once.
- **AGENT-007 — Local scheduling is honest.** When agents share one local model,
  Krill states that generation is serialized even though sessions are
  independently steerable. True concurrent decode requires separate engines or
  a batching design that has been verified for the selected model.

### Permissions and safety

- **SAFE-001 — Permission parity.** Voice cannot bypass the active Krill
  permission posture.
- **SAFE-002 — Explicit risky approvals.** Destructive, secret-bearing, or broad
  external actions require a visible confirmation. Ambient speech alone cannot
  approve them.
- **SAFE-003 — Interrupt and cancel.** The user can stop TTS, cancel capture, and
  cancel agent work independently.
- **SAFE-004 — No unlicensed impersonation.** Custom voices require a user-owned
  recording or explicit permission from the speaker/rightsholder. Scraping a
  convenient online video is not an accepted enrollment workflow.
- **SAFE-005 — Synthetic disclosure.** Product UI identifies generated speech
  as synthetic; exported clips retain model-license attribution or watermarking
  requirements.

### Native/local architecture

- **LOCAL-001 — No Python runtime.** Shipping voice mode uses Swift plus Apple
  frameworks, Core ML, or MLX Swift.
- **LOCAL-002 — Offline default.** After an explicitly consented model download,
  capture, transcription, agent work, and synthesis can run without a network.
- **LOCAL-003 — Resource separation.** Prefer Core ML/Apple Neural Engine for
  continuous audio models and MLX/Metal for the coding model so always-on voice
  does not monopolize the GPU.
- **LOCAL-004 — Pluggable engines.** STT, endpointing, and TTS sit behind small
  protocols. Apple system engines remain zero-download fallbacks.
- **LOCAL-005 — Download transparency.** Show model name, license, approximate
  size, language coverage, and whether voice cloning is supported before a
  download.

## Model posture

Krill should not train a speech model for this feature.

- **Zero-download prototype:** Apple on-device Speech plus an explicitly chosen
  enhanced/premium `AVSpeechSynthesisVoice`.
- **Natural built-in product voice:** Kokoro 82M through a native Swift/Core ML
  integration such as [FluidAudio](https://github.com/FluidInference/FluidAudio),
  or a direct MLX Swift port. Kokoro is compact and has preset voices but does
  not clone arbitrary speakers.
- **Optional consent-based custom voice:** streaming PocketTTS through
  FluidAudio. Enrollment accepts a clean, user-owned short reference recording
  and stores the derived voice material locally. The shipped model's
  attribution terms must be included.
- **Smallest experimental pack:** KittenTTS is a useful size benchmark, but its
  developer-preview status and ONNX integration make it a later experiment.

Python-first cloning systems may be used for isolated quality evaluation, but
not as the production Krill dependency.

## Prototype definition

The first reviewable prototype proves the riskiest product seams without
pretending to be full duplex:

- Native always-on-top AppKit panel launched from `krill code --voice`.
- Click-to-start/click-to-stop capture using the existing microphone recorder.
- Apple on-device transcription using the existing recognizer.
- The transcript is sent through the real in-process `AgentLoop` and local MLX
  model.
- Live agent/tool events appear in the panel.
- The final answer is spoken locally and remains visible.
- Ask-mode tool calls receive a native visual approval dialog.
- Closing the panel stops capture, speech, and in-flight work.

Global hotkeys, learned EOU/VAD, subagent cards, result collection, screen
context, and downloadable neural TTS remain after the current review milestone.

## Success criteria

The feature is ready for a production claim when a user can conduct a ten-minute
coding session without touching the terminal, interrupt Krill reliably, delegate
and redirect at least two tasks, understand which agent is blocked, approve
actions safely, and hear a natural response with no cloud or Python process.
