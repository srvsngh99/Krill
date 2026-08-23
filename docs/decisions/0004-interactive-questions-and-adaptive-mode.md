# 0004. Interactive questions, plan promotion, and the adaptive posture

Status: adopted. Date: 2026-08-23. Owner: Sourav. Scope: how Krill's agent asks
the user a question mid-run, how it leaves plan mode, and the new posture in
which it decides that for itself.

---

## 0. TL;DR

- **One seam, two tools.** A single `UserQuestionGate` — the structural twin of
  `PermissionGate`, but returning structured data instead of a `Bool` — backs
  both `ask_user` and `request_execute`. This mirrors what opencode ships (its
  `question` and `plan_exit` tools are two callers of one `ask` service), and
  avoids a second gate protocol × three implementations × three surfaces.
- **Both tools are `isReadOnly = true`.** `PermissionPolicy.decision` resolves
  read-only *before* it consults the mode (`Permission.swift:130`, ahead of the
  switch at `:131`), so this makes them available in every posture with **no
  policy carve-out**. For `request_execute` it is load-bearing: a mutating
  version would be denied by the very mode it exists to exit.
- **One question per call, flat schema.** Not opencode's nested tabbed
  multi-question payload — the grammar repair pass is capped at 256 tokens, so a
  repair regeneration of a nested payload truncates mid-JSON and is rejected.
- **`PermissionBox` owns the posture**, and its `promote()` is guarded so an
  agent can only ever move *from* a planning posture *to* `.ask`/`.acceptEdits`.
  An agent can never grant itself `.acceptAll`.
- **New `.adaptive` posture**: starts read-only, and the model promotes itself
  with no prompt. `"auto"` keeps meaning `.acceptAll` — re-pointing it would
  silently reinterpret every existing config.
- **Loosening is the model's request; tightening is the human's prerogative.**
  The model has no demotion path; the human gets Shift+Tab (which is inert
  mid-turn today and must be fixed).

---

## Context

Krill's agent could not ask the user anything. Of its 14 tools
(`Sources/KrillHarness/Tools/`) none takes user input, and the only
human-in-the-loop seam in `AgentLoop` is `:283` → `await gate?.approve(...)`,
whose protocol (`Permission.swift:149-151`) is `(String, String) -> Bool` —
binary in, binary out, and only ever triggered as a side effect of the model
calling a *mutating* tool. The model could not initiate it, attach a question to
it, or receive anything back but yes/no.

Two prompt fragments actively suppressed asking: `AgentEnvironment.swift:12`
(*"Once you have the tool results, reply with the final answer"*) and the plan
steer (*"present a clear, concise step-by-step plan as your final answer"*). The
string `clarif` appeared nowhere in `Sources/`.

Plan mode had no exit. `plan` was purely a permission posture — mutating tools
denied (`Permission.swift:138-141`) — with no `ExitPlanMode` equivalent. The
model simply stopped calling tools, `AgentLoop.swift:210` called that a final
answer, and the plan was inert prose that nothing parsed, stored, or gated on.
The posture changed only when the user pressed Shift+Tab.

The prompt for this work was a side-by-side against opencode 1.18.21 on the same
task: it asked four scoped questions before building; Krill emitted a plan with
no way to steer it.

## Decision

**1. `UserQuestionGate` — one seam.** `Sources/KrillHarness/UserQuestion.swift`
defines `UserQuestion {header, question, body, options}`,
`UserAnswer {text, optionIndex, wasFreeText, declined}`, and a protocol with
`ask(_:) async -> UserAnswer` plus `cancelPending()`. Three implementations, one
per surface, mirroring the three `PermissionGate` implementations exactly.

`optionIndex` is carried rather than matching option *labels* as strings, because
`request_execute` maps the chosen option to a `PermissionMode` and string
matching would be brittle. Every gate knows the index natively.

**2. `ask_user`** — flat schema, one question, `options` as `[String]`, injected
gate, `isReadOnly = true`. Parses defensively (accepts `[String]`,
`[{label|text|…}]`, or a delimited string) because repair cannot enforce shape —
see "Why" below.

**3. `request_execute`** — the model *requests a promotion*; it does not "finish
a plan". Takes a one-line `summary` (not a plan file — see alternatives). Three
run-time arms keyed on posture: `.plan` asks the user with three options;
adaptive self-promotes with no prompt; anything else is a benign no-op.

**4. `PermissionBox`** (`Sources/KrillHarness/PermissionBox.swift`) is the single
source of truth for a run's posture. It carries an immutable `origin` (the
posture the *user* chose — the identity the chip reports) alongside the mutable
effective policy, and exposes two mutators with deliberately different powers:

```swift
/// MODEL-facing. Only ever moves a PLANNING posture to one an agent may grant
/// itself. Returns false and changes nothing otherwise.
@discardableResult func promote(to m: PermissionMode) -> Bool {
    guard _policy.mode == .plan || _policy.mode == .adaptive else { return false }
    guard m == .ask || m == .acceptEdits else { return false }
    _policy = PermissionPolicy(mode: m, allow: _policy.allow, deny: _policy.deny)
    return true
}
/// HUMAN-facing. Any posture, any direction.
func setPolicy(mode: PermissionMode)
```

`AgentLoop` gains an optional `permissionBox` (defaulted `nil`, so every existing
call site compiles unchanged) and reads it *inside* the per-call loop at
`:272-273`, which is what makes a flip effective on the very next tool call in
the same turn. The loop relays a declared `ToolResult.effect` as an event for
observability; it never learns a tool name.

**5. `.adaptive`** — a fifth `PermissionMode` that starts read-only
(`initialEffective == .plan`) and promotes itself to
`executePosture == .acceptEdits`. `"auto"` continues to parse to `.acceptAll`.

## Alternatives considered

**A second gate protocol for plan approval.** Three implementations × three
surfaces, a second SSE frame pair, a second HTTP route, and a second TUI modal —
for a prompt that is literally "a question with three labelled options".
Rejected: opencode demonstrates the one-service shape works, and the reuse here
is total (`QuestionPrompt`, all three askers, the `/questions` route and the web
card are shared verbatim).

**opencode's nested multi-question schema** (`{questions: [{question, header,
custom, options: [{label, description}], multiple}]}`), rendered as tabs.
Rejected on a hard constraint: the grammar repair pass is capped at 256 tokens
(`EngineGenerator.swift:71`, `AgentSessions.swift:37`), so a repair regeneration
of that payload truncates mid-JSON and is rejected, leaving the model stuck with
its original bad args and no recovery. A 12B emitting the whole nested object
inside a `<tool_call>` sentinel in one shot is the failure case, not the
exception.

**Trusting the repair pass to fix argument shape.**
`ToolCalling.argsSatisfySchema` (`:357-366`) is
`required.allSatisfy { args[$0] != nil }` — presence only, never types. So
`{"question":"x","options":"a, b, c"}` *passes* and repair never fires. Rejected:
the tool must parse defensively instead.

**A plan file, as opencode uses.** Its `plan_exit` takes no arguments because the
plan lives in a file whose path the approval question interpolates. Rejected for
Krill on three grounds: writing a plan file *is a write*, which plan mode denies,
so it would need a carve-out for exactly one path in the cleanest part of the
permission model; Krill's loop continues with the same message array (no second
agent with a fresh context needs to read it back); and it litters the user's repo
with artefacts they did not ask for. A one-line `summary` argument closes the
real gap — that the plan may have scrolled away in the TUI, and is unreadable in
a phone card.

**Injecting a synthetic user message on approval,** as opencode does. Rejected as
redundant: Krill already delivers *every* tool result as a `user`-role turn
(`AgentLoop.swift:250-259`). opencode needs two channels because its tool return
and its message injection are separate; Krill's are the same channel. Appending a
second turn would produce two consecutive `user` messages and break the
template-safety invariant documented at `:246-249`.

**opencode's binary Yes/No on plan approval.** Rejected: that is a limitation of
its two-agent model (plan agent / build agent), not a design. Krill has one agent
with four postures, and `.acceptEdits` is explicitly documented as "the middle
posture" (`Permission.swift:15-20`). Offering three options costs nothing —
`QuestionPrompt` renders N — and discarding the distinction would throw away
expressiveness Krill already has.

**Offering `.acceptAll` among the approval options.** Rejected on safety: see
below.

**A `hasPromoted` latch to prevent thrashing.** Rejected: if a human demotes
mid-run, the agent may legitimately ask again, and a latch wedges that. The
"already implementing" branch tests the *current effective mode* instead.

**A demotion tool (`direction: "plan"` or a second `enter_plan_mode`).**
Considered and rejected — see below.

**Naming the new mode `auto`,** matching the user's phrasing. Rejected: a
three-way collision. `Permission.swift:30` maps `"auto"` → `.acceptAll`, `:62`
makes `.acceptAll.label` the literal `"auto"`, and `WebUI.swift:461`,
`docs/TUI.md:40,:121` and `docs/AGENT_UI.md:121` all document it that way.
Re-pointing it would silently reinterpret every existing
`default_agent_permissions = "auto"` and every `--permission-mode auto`
invocation, with no error and no migration — precisely what `configuredDefault`
exists to prevent. `.solo` and `.pilot` were also rejected: both describe
*autonomy level*, which is what `.acceptAll` already is. `.adaptive` names the
actual distinguishing property — the leash changes during the run.

**Phase as a field on `PermissionPolicy`, or as `.adaptivePlanning` /
`.adaptiveExecuting` cases.** Rejected: the first breaks the type's documented
contract as a pure, side-effect-free value type (`Permission.swift:92-95`); the
second doubles every switch and would let a user hand-select or persist
`adaptive-executing`, which is meaningless as a *starting* posture. Phase lives
on `PermissionBox` instead, derived from `origin` plus the effective mode.

**A shared box vs. a declared `ToolResult.effect` with a loop-local `var
policy`.** Both were designed. Resolved in favour of the box owning the write,
because `promote()`'s guard is a real safety mechanism that only works if the box
is authoritative; the effect remains as the observability signal.

## Why this over the others

**Read-only is the honest classification, and it pays twice.** `Tool.swift:26-28`
defines `isReadOnly` as "only observes (never writes files or runs commands)".
`DispatchTool` — which spawns an entire background agent — is already marked
read-only on exactly this reasoning (`:30-33`), as is `TodoTool` (`:9-11`).
Neither new tool touches the filesystem or runs anything. Because
`PermissionPolicy.decision` resolves read-only *above* the mode switch, this
single fact satisfies the "asking works in every posture" requirement with zero
policy changes, and keeps `deny: ["ask_user"]` working as a real kill switch
(deny still has highest precedence).

The decisive argument is a reductio in each direction: a mutating `ask_user`
would produce an absurd double prompt in `.ask` mode (`Allow ask_user(...)?
[y/n]` followed by the question itself), and a mutating `request_execute` would
be **denied by the very mode it exists to exit**.

**Safety: promotion lands on `.acceptEdits`, never `.acceptAll`.** Adaptive means
the agent decides *when* to switch phase, not *how much blast radius* to take on.
This is not a hollow distinction here — `EditTool.swift:9`,
`MultiEditTool.swift:9` and `WriteTool.swift:8` all set `isFileEdit = true` while
`BashTool` does not, so `Permission.swift:139` genuinely unblocks the productive
edit loop while routing every shell command to the human. Edits are recoverable
via git; `rm -rf` and `curl | sh` are not. Full autonomy stays reachable by a
deliberate Shift+Tab — an explicit act, never a one-keystroke answer to a
*model-initiated* prompt. Enforced structurally by `promote()`'s guard rather
than by convention.

**Asymmetry on demotion.** Loosening a leash is the model's request; tightening
is the human's prerogative. Giving a model both directions gives it a knob to
oscillate on, and the runaway guard cannot stop it — that guard keys on
`name + argumentsJSON` (`AgentLoop.swift:309-311`), so a model that merely
*rewords* its `summary` produces a fresh signature every time. The downside of
never demoting is nil: the agent holds a permission it is not using. Making
`promote()` a one-way state machine dissolves the thrash problem instead of
guarding against it.

**One question per call is a constraint, not a preference.** See the 256-token
cap above. If the model needs two questions it calls twice; consecutive distinct
calls have different signatures and pass the dedupe guard.

## Consequences

**For users**
- The agent asks before guessing on genuinely ambiguous work, in every posture.
- Plan mode gains a real exit: the agent presents a plan and asks to start, with
  a choice of how much leash to grant.
- A new `adaptive` posture where the agent moves itself from planning to editing
  without interruption — while shell commands still prompt.
- Self-promotion is silent to the *interaction* but loud in the *surfaces*: a
  transcript note, a chip that flips `adaptive (planning)` → `adaptive
  (executing)`, and a one-shot status line. A promotion the user cannot see is
  the failure mode that would destroy trust in this feature.

**For operators / config**
- `"auto"` is unchanged. `"adaptive"` is new.
- Forward-compat: an older Krill reading `default_agent_permissions = "adaptive"`
  gets `parse` → nil → `configuredDefault` → `.plan`. Incompatible in the safe
  direction.
- New route `POST /v1/agent/sessions/{id}/questions` and three new SSE frames.

**Costs and known sharp edges**
- `ChatTUI.agentTools()` becomes `agentTools(asker:)`, and each background
  `AgentSession` needs its own asker and `PermissionBox` — a shared asker would
  let one session overwrite another's parked continuation.
- The live widget must be driven by `asker.pending()` on the render tick, **not**
  by an `AgentEvent`: the loop emits `toolFinished` only after `run` returns, so
  an event-driven widget structurally cannot appear while the tool is parked.
- The tool-system turn is frozen after `ensureAgentSeed()`
  (`ChatTUI.swift:1336-1338`) while Shift+Tab changes posture freely, so
  posture-conditional tool *registration* is a lie in the TUI. Both tools
  register unconditionally there and branch at run time. This is a pre-existing
  soundness gap for any posture-dependent toolset; tracked in `docs/BACKLOG.md`.

**Pre-existing bugs this work forces us to fix** (each is a real defect
independent of the feature):
- `CodeCommand.swift:220` wires `gate: nil` outside ask/accept-edits, so after
  any promotion every `bash` call is silently denied forever with no prompt
  printed.
- `RemoteApprover.approve` parks forever when no client is attached.
- Shift+Tab is inert mid-turn (`handleAgentRunKey` does not handle `.backTab`),
  so there is currently no way to re-leash a running agent.
- `complete(messages:constrainingToolNames:)` is extension-only, so the tool-name
  logit mask — "layer 0" — has never run in production.
- `GenerationStats` is published after the terminal `isEnd` on the generic dense
  path, so agent turns never record stats and the footer never shows context use.

## Testing / verification

Full matrix in `docs/ASK_USER_VERIFICATION.md`. The assertions that matter most:

- **`promote(to: .acceptAll)` returns `false` and changes nothing.** The single
  most important safety assertion in this work.
- **`Set(PermissionMode.allCases.map(\.label)).count == 5`** — the test that
  would have caught the `"auto"` collision before it shipped.
- **`parse("auto") == .acceptAll`** — pins the config-compat guarantee.
- **`PermissionPolicy(mode: .executePosture).decision("bash", isReadOnly: false)
  == .ask`** — promotion never hands over unprompted shell.
- For **all five** postures, `decision("ask_user", isReadOnly: true) == .allow`.
- An `AgentLoop` end-to-end test: `request_execute` then `write_file` from a
  `.plan` start, asserting the write is **not** denied and that `allow`/`deny`
  survived the flip. Only this proves the decision site re-reads the box mid-run.
- The adaptive path invokes the question gate **zero** times.
- A declined question/plan is `isError: false` — an error result pushes small
  models into a retry loop.
- Timeout tests assert `cancelPending()` ran: an unresumed `CheckedContinuation`
  both leaks and wedges the gate for the rest of the session.

All tests live in library targets (`KrillHarnessTests`, `KrillTUITests`,
`KrillServerTests`, `KrillEngineTests`) — never `KrillCLI`, whose `@main`
breaks the swift-testing pass. This is why `QuestionPrompt` lives in `KrillTUI`
and `PermissionBox.chipLabel` in `KrillHarness` rather than inline in `ChatTUI`.
