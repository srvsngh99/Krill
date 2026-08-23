# Krill: interactive questions, plan approval, adaptive mode, and a TUI lift

Status: adopted. Date: 2026-08-23. Scope: giving Krill's agent the ability to
ask the user questions, to request promotion out of plan mode, a new adaptive
posture, real file diffs, and the TUI chrome to surface all of it.

> **This is the SPEC — why and what.** Three companion docs complete the set:
>
> | Doc | Purpose |
> | --- | --- |
> | `docs/ASK_USER_AND_ADAPTIVE_MODE.md` (this) | Design, rationale, verified gotchas |
> | `docs/ASK_USER_IMPLEMENTATION.md` | Ordered work plan — files, edits, acceptance per step |
> | `docs/ASK_USER_VERIFICATION.md` | Test matrix and manual QA script |
> | `docs/decisions/0004-interactive-questions-and-adaptive-mode.md` | ADR — decisions and rejected alternatives |
>
> Everything below was verified against the source at commit `973c885`; line
> numbers are accurate as of that commit and should be re-checked if the branch
> has moved. Claims marked *(verified)* were confirmed by reading the code
> directly, not inferred. **Implemented and adopted on `feature/krill-ui`.**

## Context

Comparing Krill's TUI against opencode on the same prompt exposed a capability
gap, not a prompting or model-quality gap.

**1. Krill's agent can never ask a question.** Krill exposes 14 tools
(`Sources/KrillHarness/Tools/`) and **none can ask the user anything**. The only
human-in-the-loop seam in the entire agent loop is `AgentLoop.swift:283` →
`await gate?.approve(...)`, whose protocol (`Permission.swift:149-151`) is
`(String, String) -> Bool`. Binary in, binary out. The model cannot initiate it,
cannot attach a question to it, and cannot receive anything back but yes/no.

**2. Krill's prompts push the opposite way.** Only four prompt fragments exist in
the whole agent stack, and two terminate the turn in an answer: *"Once you have
the tool results, reply with the final answer"* (`AgentEnvironment.swift:12`) and
*"present a clear, concise step-by-step plan as your final answer"*
(`CodeCommand.swift:155`). The string `clarif` appears nowhere in `Sources/`.

**3. Plan mode has no approval gate.** `plan` is only a permission posture:
`Permission.swift:138-141` denies mutating tools, full stop. There is no
`ExitPlanMode` equivalent — the model stops calling tools, `AgentLoop.swift:210`
calls that a final answer, and the plan is inert prose. The posture changes only
when the user presses Shift+Tab.

Model size (gemma-4-12b vs opencode's frontier model) is a real but fourth-order
factor — it affects how *reliably* a question gets asked, not whether asking is
possible.

Two further items surfaced during review and are folded in: the footer never
shows context *consumed* (Part D — a genuine bug, plus four others found with
it), and the chrome is visibly less finished than opencode's (Part C).

## Scope (confirmed with user)

- Ask-tool, plan-approval gate, **adaptive posture**, live task/progress panel,
  and the context-usage bug.
- All three surfaces: full-screen TUI, `krill code` CLI, remote/web sessions.
- **Asking works in every posture** — "building it right is more important."
- **The agent may request promotion** to an executing posture.
- **A new adaptive posture** where the model decides for itself when it is
  planning vs executing.

---

## Reference architecture (extracted from the shipped opencode binary)

Decompiling `/opt/homebrew/Cellar/opencode/1.18.21/bin/opencode` gives the shape
to mirror. The key discovery: **`question` and `plan_exit` are two tools over ONE
`ask` service.** We should not build two mechanisms.

`question` takes `{questions: [Prompt]}` where `Prompt` is
`{question, header, custom, options: [{label, description}], multiple?}`, and
returns `"<question>"="<label>"` per question. `plan_exit` takes `{}` and calls
the *same* ask service with one synthesized Yes/No, then switches agent to
`build` and injects a synthetic user message.

Four things **do not port** and should not be copied:
1. **The empty `plan_exit` schema.** opencode's plan lives in a *file* it names;
   Krill has none, so an empty-args tool gives the approval widget nothing to
   show. Krill takes a `summary` argument instead.
2. **`RejectedError` on decline.** An `isError: true` result reads to a 12B as
   "the tool broke, retry."
3. **The synthetic user message.** Krill already delivers *every* tool result as
   a `user` turn (`AgentLoop.swift:250-259`) — opencode needs two channels
   because its tool return and message injection are separate. Ours are the same.
4. **"Switch to the build agent."** Krill has one loop and one history carried by
   `priorMessages`, not two agents. The analogue is *only* the posture flip.

What **does** port is opencode's much richer plan-mode steering (~200 words, with
an explicit instruction to ask when weighing tradeoffs) versus Krill's single
sentence, and its plan→build mode-change reminder, which Krill lacks entirely.

---

## Cross-cutting gotchas (verified — must not be missed)

- **`PermissionPolicy.decision` allows read-only tools BEFORE consulting the
  mode** (`Permission.swift:130`, ahead of the `switch` at `:131`). So a tool
  declaring `isReadOnly = true` is permitted in *every* posture including
  `.plan`. Both new tools rely on this; **no policy carve-out is needed, and
  adding one would be wrong.**
- **`krill code` wires no gate in plan mode** —
  `gate: (mode == .ask || mode == .acceptEdits) ? StdinApprover() : nil`
  (`CodeCommand.swift:220`). This is a **latent deadlock**, not just a wiring
  gap: after a promotion to accept-edits, every `bash` call yields `.ask`, hits
  `gate?.approve(...) ?? false`, and is silently denied forever. Fix:
  `gate: RawTerminal.isInteractive ? StdinApprover() : nil` — inert in
  `.plan`/`.acceptAll` (never consulted today), correct after promotion.
- **The tool-system turn is frozen after `ensureAgentSeed()`**
  (`ChatTUI.swift:1336-1338`; re-injection skipped at `AgentLoop.swift:157-160`)
  while Shift+Tab changes posture freely. Posture-conditional tool registration
  is therefore a lie in the TUI. Register unconditionally; branch at run time.
  Worth a follow-up issue — this makes *any* posture-dependent toolset unsound
  after the first Shift+Tab.
- **The grammar repair pass is capped at 256 tokens** (`EngineGenerator.swift:71`,
  `AgentSessions.swift:37`), and **`argsSatisfySchema` checks only *presence* of
  required top-level keys, never types** (`ToolCalling.swift:357-366`). Together
  these rule out nested multi-question schemas and mean tools must parse
  defensively rather than trust repair.
- **Tool results ride the `user` role**, not a `tool` role
  (`AgentLoop.swift:250-259`). Never append a second synthetic `user` turn — two
  consecutive user turns break the template-safety invariant documented at
  `:246-249`.
- **The cross-task hazard resolves itself.** `runAgentTurn`'s `onEvent` pushes
  into `EventQueue` (`ChatTUI.swift:1367`) and the **main task** drains it via
  `applyAgentEvent` (`:1373, :1528`). A mode-change event therefore arrives on
  the main task already — no lock, no `@MainActor`, no new machinery. State this
  in the PR so nobody over-engineers it.
- Per repo memory: **test targets must not depend on the executable target** —
  use library targets (`KrillHarness`, `KrillTUI`, `KrillServer`), never
  `KrillCLI`.

---

## Part D — the context bug (land FIRST)

**It is a race, not a missing feature.** The rendering code at
`ChatTUI.swift:2079-2095` is correct and complete; `lastStats` is simply always
nil, so it falls through to `:2091-2092` and prints the bare `ctx 128K`.

**Root cause.** `InferenceEngine.generate(messages:)` — the generic dense path
gemma-4-12b uses — publishes `GenerationStats` at `InferenceEngine.swift:2021`,
*after* every terminal `isEnd` yield (`:1787, :1805, :1863, :1893, :1929, :1955,
:1998`). `EngineGenerator.collect` breaks the instant it sees `isEnd`
(`EngineGenerator.swift:86`) and reads `stats()` at `:35` with **zero intervening
work**, so it reads a still-nil holder and `onStats` never fires.

This inverts a contract the codebase already documents: the five VLM drivers
publish stats *before* the terminal event, with the comment at
`InferenceEngine.swift:2269-2281` — *"setting stats first keeps the
(stream, stats) contract race-free."* The generic dense path is the **only**
producer that gets it backwards. Every other consumer survives by accident:
`RunCommand` (`:406-409`), `Server` (`:1626-1639`) and plain-chat `ChatTUI`
(`:1997-2029`) all do real work between the break and the read.

Aggravating factor: agent turns are `.greedy` (`EngineGenerator.swift:32`), which
enables the n-gram speculative path where `isEnd` lands microseconds behind the
content tokens — so the consumer is already awake and loses the race every time.

*(An earlier hypothesis that stats are emitted only on truncation was tested and
refuted: `:1998` is not the only terminal event, and `:2021` is reached on both
the EOS and truncation paths.)*

**Fix:** publish stats before the terminal yield, matching the documented VLM
ordering. Regression test: a consumer that breaks on the first `isEnd` and
immediately reads `stats()` observes non-nil.

**Four further defects found in the same trace:**

1. **No final drain of `AgentStatsBox`.** `ChatTUI.swift:1380` is the only
   `take()`; the loop breaks at `:1392` and `:1399-1404` never re-checks, so a
   `put` landing between `:1380` and `:1382` is lost permanently. Add a final
   `take()` after the loop.
2. **The tool-name logit mask has never run in production.** *(Verified
   directly.)* `completeConstrained(messages:jsonSchema:)` **is** a protocol
   requirement (`HarnessGenerator.swift:27`), so it dispatches dynamically and
   both repair passes work. But `complete(messages:constrainingToolNames:)`
   exists **only in the extension** (`:45-49`), never as a requirement — so the
   call at `AgentLoop.swift:199` on the existential `any HarnessGenerator`
   (`:94`) statically dispatches to the default, which ignores the constraint.
   `EngineGenerator.swift:43-63`, described in its own comments as "layer 0", the
   primary defence against a model naming an unoffered tool, is dead code.
   **Fix: promote it to a protocol requirement.**
3. **Three `EngineGenerator` sites never set `onStats`**, so they structurally
   cannot show tok/s: `CodeCommand.swift:215`, `AgentSession.swift:56`,
   `ChatTUI.swift:1422` (`/research`).
4. **`processSubmit` routes to `activeSession` (`ChatTUI.swift:821-825`) before
   the `surface == .agent` check (`:838`)**, so while attached to a background
   session every turn uses the no-`onStats` generator from defect 3.

`/context` (`ChatTUI.swift:1188-1192`) inherits the same nil and **silently
presents a ~4-chars/token estimate as if it were measured**. Fixing the root
cause fixes footer, `/context`, and the Part C sidebar together — which is why
Part D lands first, or the new panel just displays wrong numbers in a nicer box.

---

## Part A — `ask_user`

**New seam** `Sources/KrillHarness/UserQuestion.swift`:

```swift
public struct UserQuestion { let header, question, body: String; let options: [String] }
public struct UserAnswer   { let text: String; let optionIndex: Int?
                             let wasFreeText, declined: Bool }
public protocol UserQuestionGate: Sendable {
    func ask(_ q: UserQuestion) async -> UserAnswer
    func cancelPending()            // MUST be safe when nothing is pending
}
```

Shaped on `PermissionGate` (one request in flight, parks on a continuation) but
returning structured data. `optionIndex` and `body` exist because Part B needs
them — mapping option *labels* to postures by string match would be brittle, and
every gate impl already knows the index natively. `cancelPending()` on every path
(timeout, Ctrl-C, teardown): an unresumed `CheckedContinuation` both leaks and
wedges the gate for the rest of the session.

**New tool** `Tools/AskUserTool.swift`, name `ask_user` (snake_case verb-object,
matching `read_file`/`web_search`). Aliases `question`/`askuserquestion`/`ask` →
`ask_user` in `ToolCalling.swift:1394-1413`.

**Schema is flat — ONE question, `options` as `[String]`.** This departs from
opencode's nested tabbed payload because of the 256-token repair cap: a repair
regeneration of a nested payload truncates mid-JSON and is rejected, leaving the
model stuck with its original bad args. And since `argsSatisfySchema` is
presence-only, repair never even fires for `{"options": "a, b, c"}`. **So the
tool must parse defensively**: accept `options` as `[String]`, as
`[{label|text|title|value}]`, or as a single delimited string; fall back
`question` → `prompt`/`text`; dedupe; cap at 6.

**`isReadOnly = true`.** Precedent: `DispatchTool` (which spawns a whole
background agent) and `TodoTool` are both read-only on the same reasoning. The
decisive argument against making it mutating: in `.ask` mode it would produce an
absurd double prompt — `Allow ask_user(...)? [y/n]` followed by the question
itself. Read-only also preserves `deny: ["ask_user"]` as a real kill switch,
since the deny list still outranks everything.

**The gate is injected into the TOOL, not the loop.** `AgentLoop` has zero
tool-specific knowledge and should keep it; `PermissionGate` is a loop concern
because it gates *every* tool, whereas asking is one tool's capability — the
`DispatchTool(queue:)` / per-session `todoTool` pattern. Cost: `agentTools()`
becomes `agentTools(asker:)`, and **each background `AgentSession` needs its own
asker** (they currently share one registry, `ChatTUI.swift:1594-1596`; a shared
asker would let one session overwrite another's parked continuation).

**The live widget is driven by `asker.pending()` on the render tick, NOT an
`AgentEvent`.** The widget must be on screen *while `AskUserTool.run` is parked*,
but the loop emits `toolFinished` only after `run` returns
(`AgentLoop.swift:250-259, :288`) — so an event-driven widget structurally cannot
appear in time. This is exactly how the existing approval prompt works
(`ChatTUI.swift:2508-2513`). **Cache the widget keyed on the `UserQuestion`
value**, not merely on nil→non-nil, or a back-to-back `ask_user` then
`request_execute` in one turn renders the stale widget.

Widget: new pure struct `Sources/KrillTUI/QuestionPrompt.swift` modelled on
`AgentPicker`, rendered by `renderQuestion` beside `renderPicker`
(`ChatTUI.swift:763`), keys in `handleAgentRunKey` (`:1505`) *before* the
approver branch: ↑/↓, 1-9, Enter, `t` to type, Esc to skip. If free-text proves
fiddly, **ship options + Esc-to-skip first** — the result already distinguishes
the two cases.

**Degradation:** `krill code` with piped stdin **does not register the tool at
all**, so the model is never told it exists — no hang possible, no context
wasted. Elsewhere: Esc/blank/EOF → declined, plus a 300s timeout calling
`cancelPending()`. Declined is `isError: false` (an error pushes the model into a
retry loop) and says *"state the assumption you made, and do not ask again."*

Do **not** revive `OperatorEvent.confirmationNeeded` — it belongs to
`KrillAgent`'s unrelated `OperatorLoop`, and `KrillHarness` does not depend on
`KrillAgent`.

---

## Part B — `request_execute` + the adaptive posture

**One seam, two tools.** `UserQuestionGate`, all three asker impls, the
`/questions` route, the WebUI card, `QuestionPrompt` and `renderQuestion` are
reused **100% unchanged**. Part B's only surface work is displaying the *flip*,
not the prompt.

**Name: `request_execute`.** Framing matters — the model *requests a promotion*,
it does not "finish a plan". Aliases `exitplanmode` (Claude Code's published name
that fine-tunes emit), `plan_exit`, `exitplan`, `approve_plan`,
`start_implementing`.

**No plan file.** opencode needs one because its *build* agent has a fresh
context and must read the plan back; Krill's loop continues with the same message
array, so the plan is already verbatim in the transcript. Writing a plan file in
plan mode would also be a *write in read-only mode*, needing yet another
carve-out. Instead take a one-line `summary` argument, shown in the widget below
the plan already on screen. Schema:

```json
{"type":"object","properties":{
"summary":{"type":"string","description":"One sentence naming what you will do if approved."}},
"required":["summary"]}
```

`summary` is required so presence-only checking catches the empty-`{}` call weak
models produce. Demotion is deliberately absent: tightening the posture remains
a human-only action through Shift+Tab.

**`isReadOnly = true` is load-bearing**, by reductio: a mutating
`request_execute` would be **denied by the very mode it exists to exit**. Pin it
in a test.

**Registration: always, with a posture-aware early return.** Conditional
registration is a lie in the TUI (frozen tool-system turn — see gotchas). Three
run-time arms:

| effective posture | behaviour |
|---|---|
| `.plan` | ask the user (3 options), flip on approval |
| adaptive-derived `.plan` | **self-promote, no prompt**, announce it |
| `.ask` / `.acceptEdits` / `.acceptAll` | no-op: *"You already have edit permission. Just do the work."* |

### The flip: `PermissionBox` is authoritative; the event is observability

*(Two design passes disagreed here — one proposed a shared box, one a declared
`ToolResult.effect` with a loop-local `var policy`. Reconciled as follows, because
the box's `promote()` guard below is a real safety mechanism that only works if
the box owns the write.)*

New `Sources/KrillHarness/PermissionBox.swift` — a lock-guarded holder that is the
**single source of truth** for a run's posture:

```swift
public final class PermissionBox: @unchecked Sendable {
    /// The posture the USER chose. Never changes for the life of the run — it is
    /// the identity the chip and web UI report. An adaptive run stays `.adaptive`
    /// even while executing; the PHASE derives from the effective policy.
    public let origin: PermissionMode
    public var policy: PermissionPolicy { /* locked */ }
    public var effective: PermissionMode { policy.mode }
    public var isPlanning: Bool { effective == .plan || effective == .adaptive }

    /// Pure, testable chip string — lives here, not in ChatTUI, so the one
    /// user-visible requirement (mode AND phase) is unit-testable at all.
    public var chipLabel: String {
        origin == .adaptive ? "adaptive (\(isPlanning ? "planning" : "executing"))"
                            : origin.label
    }

    /// MODEL-facing. Only ever moves a PLANNING posture to one an agent may
    /// grant itself. Returns false and changes nothing otherwise — this guard
    /// is what keeps `.acceptAll` permanently off the table.
    @discardableResult public func promote(to m: PermissionMode) -> Bool {
        guard _policy.mode == .plan || _policy.mode == .adaptive else { return false }
        guard m == .ask || m == .acceptEdits else { return false }
        _policy = PermissionPolicy(mode: m, allow: _policy.allow, deny: _policy.deny)
        return true
    }
    /// HUMAN-facing (Shift+Tab, web UI). Any posture, any direction.
    public func setPolicy(mode: PermissionMode) { ... }
}
```

`AgentLoop` gains one optional `permissionBox` field (defaulted `nil`, so all 3
production sites and ~15 test constructions compile untouched) and reads
`(permissionBox?.policy ?? permission).decision(...)` at `:272-273` **inside** the
per-call loop — that placement is what makes a flip effective on the very next
tool call in the same turn. The loop stays tool-agnostic: it reads a box, it never
learns a tool name.

`promote()` rebuilding the policy is what makes **`allow`/`deny` survive the
flip** — critical, so `--deny-tool bash --permission-mode adaptive` still never
runs bash (deny has highest precedence, `Permission.swift:121-123`).

`ToolResult` still grows **two** optional fields in one edit to `Tool.swift:5-13`
(both defaulted `nil`): `display: ToolDisplay?` (Part A's Q&A record, Part C's
diffs) and `effect: ToolEffect?`. The effect is emitted as
`AgentEvent.permissionChanged` for **observability only** — transcript note plus
surface sync — not as the write mechanism.

**No new `AgentEntry` case.** The event folds to the **existing** `.note`, which
already renders as a dim `✳` line in the TUI, plain text in `LineAgentRenderer`,
and a `note` frame remotely. Zero new rendering. The *event* stays distinct
because surfaces need it to update their own posture state, which a note string
cannot drive.

Two `let`s must become live reads or the flip evaporates between turns:
`AgentSession.permissions` (`AgentSession.swift:16`) and `RemoteAgentSession.mode`
(`AgentSessions.swift:157` — keep it as the immutable *origin* and add the box
beside it).

### What "Yes" lands on — the prompt offers the choice

```
1. Yes — apply edits automatically, ask before commands   → .acceptEdits (default)
2. Yes — ask me before every edit and command             → .ask
3. No  — keep planning                                    → stay .plan
```

opencode's binary Yes/No is a limitation of its **two-agent** model, not a design
to copy. `.acceptAll` is **deliberately not offered**: going from read-only to
unattended shell on a single Enter — in answer to a *model-initiated* prompt — is
the one path where a mis-key is genuinely dangerous. Shift+Tab still reaches it
as a deliberate act. Map option → mode **by index**.

### The mode-change reminder

The tool result *is* the synthetic user message. The highest-leverage sentence in
the feature: *"Ignore the earlier PLAN MODE instruction in your system prompt —
it no longer applies."* On `krill code` and remote, the plan steer lives in the
**system turn**, injected once (`AgentLoop.swift:150-154`), so after the flip it
still reads *"you must NOT write files"*; a 12B needs that contradiction named.
(The TUI is accidentally better off — its steer is a per-turn user prefix.)

### Rejection

`isError: false` always — small models retry errors, and `isError: true` renders
red (`ChatTUI.styledCode:1567`); a user declining is not a failure. The text
deliberately does **not** say "call it again when ready". The runaway guard
(`AgentLoop.swift:206-208`) dedupes only *identical* `(name, argumentsJSON)`
pairs, so a reworded `summary` would sail through and re-prompt.

### The adaptive posture

**Naming collision is three-way and confirmed:** `Permission.swift:30` maps
`"auto"` → `.acceptAll`, `:62` makes `.acceptAll.label` the literal `"auto"`
(the chip reads `agent: auto`), and `WebUI.swift:461` plus `docs/TUI.md:40,:121`
and `docs/AGENT_UI.md:121` all document "auto" = accept-all.

**Name it `.adaptive`** (synonyms `adaptive`/`self`/`pilot`), and **keep
`"auto"` → `.acceptAll` exactly as-is** — remapping it would silently
reinterpret every existing `config.toml` and `--permission-mode auto`
invocation, a behaviour change nobody asked for.

```swift
case adaptive                       // rawValue "adaptive"

/// The posture a run STARTS in. `.adaptive` begins read-only and promotes
/// itself; every other mode is its own start state.
var initialEffective: PermissionMode { self == .adaptive ? .plan : self }

/// The posture `request_execute` promotes into. Edits auto-apply; commands
/// still reach the gate — the agent decides WHEN to build, not how much
/// blast radius to take on.
static let executePosture: PermissionMode = .acceptEdits
```

- `cycleOrder` (`:50`) → `[.plan, .adaptive, .ask, .acceptEdits, .acceptAll]`;
  the list is documented "safest → freest" and adaptive starts at plan's safety.
- `decision`'s switch (`:132-141`) gets a **fail-closed** `.adaptive` arm sharing
  `.plan`'s deny — policies should only ever be built from an *effective*
  posture, so reaching that arm is a wiring bug and must fail read-only.
- `configuredDefault` (`:40-47`) keeps falling back to `.plan`.

**Safety: adaptive lands on `.acceptEdits`, never `.acceptAll`.** Adaptive means
the agent decides *when* to switch phase, not *how much blast radius* to take on.
Edits are recoverable via git; `rm -rf` and `curl | sh` are not. Verified this is
not a hollow distinction here: `EditTool.swift:9`, `MultiEditTool.swift:9` and
`WriteTool.swift:8` all set `isFileEdit = true` while `BashTool` does not, so
`Permission.swift:139` genuinely unblocks the edit loop while routing every shell
command to the human. Enforced structurally by `promote()`'s guard, not just by
convention.

**Two gate-nil hazards this forces (both must be fixed):**

- **`krill code` has no gate in adaptive.** `CodeCommand.swift:220` yields `nil`
  for `mode == .adaptive`, so the instant the agent self-promotes, the first
  `bash` hits `.ask` → `gate?.approve(...) ?? false` → **denied forever, with no
  prompt printed**. Fix by constructing the approver for any posture a run can
  *reach*, not just the one it starts in:
  `gate: (mode != .acceptAll && RawTerminal.isInteractive) ? StdinApprover() : nil`.
  This also closes the same hole for Part B's plan→ask promotion.
- **Remote with no client attached hangs forever.** `.acceptEdits` +
  `RemoteApprover` parked with nobody to answer waits on a continuation
  indefinitely. **Pre-existing** for `.ask`/`.acceptEdits` sessions, but adaptive
  makes it reachable from a session created as `.adaptive` whose tab was then
  closed. Give `RemoteApprover.approve` (`AgentSessions.swift:88-92`) the same
  300s timeout `AskUserTool` gets, resolving to **deny** (fail-safe — the
  opposite polarity from a declined question, which continues). Land as its own
  commit: it is a bug adaptive forces, not part of adaptive.

**Demotion: the human can, the model cannot.** *(This supersedes an earlier
`direction: "plan"` proposal.)* The asymmetry is the point — loosening a leash is
the model's request; **tightening is the human's prerogative.** Giving a model
both directions gives it a knob to oscillate on, and the downside of never
demoting is nil (the agent merely holds a permission it isn't using). It also
dissolves the thrash problem rather than guarding against it: after promotion
`promote()` returns `false`, so a second call is a benign `isError: false` no-op
saying *"you are already implementing."*

Note the runaway guard does **not** help here — it keys on
`name + argumentsJSON` (`AgentLoop.swift:309-311`), so a model that *rewords* its
`summary` produces a fresh signature every time. The box's state machine is the
real defence. Key detail: the "already implementing" branch must test the
**current effective mode**, not a `hasPromoted` latch — if a human demotes
mid-run, the agent may legitimately ask again, and a latch would wedge that.

**Which exposes a real gap:** `handleAgentRunKey` (`ChatTUI.swift:1505-1523`)
does not handle `.backTab`, so **Shift+Tab is inert while a turn is in flight**.
Add an arm routing it to `box.setPolicy(mode:)`. That is the honest demotion
path, and it is a small independently valuable fix.

**Footer chip shows mode + phase** via `PermissionBox.chipLabel` —
`ChatTUI.swift:1787-1788` becomes `\(box.chipLabel)`, same at `:1782` for
attached background sessions. Putting it on the box rather than inline in
`ChatTUI` is what makes this requirement testable at all (`ChatTUI` is in the
executable target). The box reads under its own lock and `runAgentTurn`
re-renders every ~120ms (`:1370-1400`), so the chip follows within one tick with
no extra plumbing.

**Self-promotion is silent to the human *interaction*, but loud in every
surface.** A promotion the user cannot see is the failure mode that destroys
trust in this feature. So: the `.note` entry renders in all four consumers, the
chip flips, and `lastStatus` gets a one-shot *"agent granted itself accept-edits
· shift+tab to re-leash"*.

**The silent hazard — most likely bug in this change.** `CodeCommand.swift:196`
is an exhaustive `switch mode` and will fail the build when `.adaptive` is added
(good). But every `== .plan` comparison compiles fine and **silently skips the
plan steer for adaptive**: `ChatTUI.swift:1350`, `AgentSession.swift:50`,
`CodeCommand.swift:154`, `AgentAPI.swift:223`. This is exactly what the
prompt-hoist below fixes — the predicate then changes **once**, to
`mode.initialEffective == .plan`.

### Prompts (without these, both tools are dead code)

**Mandatory prerequisite:** the plan-mode steer has **four** verbatim copies, not
two — `CodeCommand.swift:155-160`, `AgentAPI.swift:224-229`,
`ChatTUI.swift:1349-1352`, and `AgentSession.swift:49-52` (the background-agent
copy). Hoist into `AgentEnvironment.planSystemSteer` / `planTurnPrefix` first.

1. Amend `AgentEnvironment.toolDirective` (`:11-16`) — its *"Once you have the
   tool results, reply with the final answer"* literally instructs the model to
   stop after an answer. Add the `ask_user` exception.
2. New `askUserDirective`, appended **unconditionally at all four builders**.
   ⚠️ At three of them the append site sits immediately adjacent to an
   `if mode == .plan` block (`CodeCommand.swift:154`, `AgentAPI.swift:223`,
   `ChatTUI.swift:1350`) — it must **never** land inside one.
3. Extend `planSystemSteer` to say `request_execute` is how you leave plan mode,
   and that `ask_user` is permitted — otherwise *"present a plan as your final
   answer"* reads as a prohibition on any further tool call.

---

## Part C — the TUI lift

The gap to opencode is missing structure, not taste. Most of what opencode shows,
Krill already computes but hides.

### Live task list + progress (explicit user requirement)

**1. A third task state.** `TodoTool` is two-state — `(text, done: Bool)`
(`TodoTool.swift:30`), schema `:19-27`. opencode distinguishes *active* (`[●]`,
tinted) from *pending*. **Infer `active` = first not-done item rather than adding
a `status` field.** Asking a 12B to maintain a tri-state on every call is real
failure surface for no benefit, since agents work the list top-down; accept an
explicit `active: true` if volunteered, never depend on it.

**2. Progress readout.** Sidebar header shows `N/M done`, the active step, and
turn elapsed. `lastStatus` (`ChatTUI.swift:1383-1389`) and `generatingStatus`
(`:2045-2052`) already carry spinner/elapsed/tokens but are transient, single-row
and vanish between turns. The plan-level `N/M` exists nowhere today.

**3. Fix the shared-instance bug first.** `TodoTool.swift:8` documents "each
surface keeps its own list", but `spawnSession` passes the *main* `todoTool`
through `agentTools()` (`ChatTUI.swift:1625-1630`), so every background agent
writes into the foreground list. Give each `AgentSession` its own — same
per-session fix as the asker.

**4. Expose it.** `TodoTool.items` is `private` behind an `NSLock` with no
accessor; add a lock-guarded `snapshot()`. `render()` already runs every event
tick, so polling per frame is fine.

### Layout

**No column layout exists at all.** `positioned(_:_:)` (`ChatTUI.swift:2579-2581`)
is row-addressed, always column 1; grep for `sidebar`/`paneWidth`/`leftWidth`
returns zero. The only two-column effects are left/right justification within one
row (`Brand.header`, `Brand.footer`). *The good news:* all pane content re-wraps
from source every frame (`CodeView.swift:28-31`), so narrowing the main column is
pure plumbing — no cache invalidation.

Needs: a `positioned(row:col:)` variant, a `join(left:right:width:)` row
composer, a public `visibleWidth`-aware clip (`Brand.visibleCount`/`stripAnsi`
exist but aren't in `KrillTUI`), and `mainWidth = cols - sidebarWidth - 1`
threaded into `paneLines`, `TUIMarkdown.render`, `CodeView.*`, `renderMenu`,
`inputBox`.

**Sizing rule (decided, not left open):** a fixed ~34-col right pane, shown only
when `cols >= 110`; below that suppressed entirely, with `/context` and `/status`
still covering the same information. Toggleable via key and `/sidebar`.
`updateSize()` already runs on SIGWINCH (`:1378`), so resize re-flows for free.

### What to move, not build

**The context panel already exists — it's trapped in a modal.** `/context`
(`ChatTUI.swift:1197-1322`) has a 32-cell multi-colour segmented bar
(`contextSegmentBar`), a per-category legend, `fmtTok`, and used/free/percent.
Move it into the sidebar; build nothing new.

**Session totals are computed and never shown while running.**
`sessionPromptTokens`/`sessionGeneratedTokens`/`sessionTurns`
(`ChatTUI.swift:145-151`, accumulated at `:160-166`) are read only in
`printSessionReceipt()` (`:329-353`) — which prints *after* `raw.leave()`, once
the TUI is gone.

### Polish items (each independent, ship in any order)

- **Per-block tool collapse.** `toolOutputExpanded` (`:1296`, Ctrl-O at
  `:399-405`) is one global Bool toggling every block; the collapsed placeholder
  is a bare line count with no preview (`CodeView.swift:74-78`). Needs identity
  on `Msg` (`:30-37` has none), a focus model (only `scrollOffset` exists), and a
  `preview:` parameter. **SGR mouse mode is already enabled**
  (`RawTerminal.swift:61`) and clicks already parsed but discarded
  (`KeyReader.swift:186-210`) — a literal "click to expand" is reachable.
- **Reasoning block.** `StreamingReasoningFilter`
  (`ReasoningParser.swift:235,267-274`) discards think tokens with no accessor;
  `AgentEvent` has no reasoning case. `+ Thought: 1.5s` is net-new: capture the
  text, add an event/`Msg` case, time the phase (`formatElapsed` exists).
- **Markdown.** `TUIMarkdown.swift:9-35` handles fences, ATX headings, inline
  code, bold — nothing else. **Lists get reflowed into paragraphs by
  `Layout.wrap`**, destroying structure, and task lists (`- [ ]`) are exactly the
  todo format. A second drifted renderer exists (`MarkdownStream`,
  `TerminalStyle.swift:242-308`); unify them.
- **Palette.** `Palette` (`KrillTUI/Theme.swift:15-29`) has four slots; the
  5-stop ember spectrum is duplicated in three places (`TerminalStyle.swift:45`,
  `:173-176`, `ChatTUI.swift:2070-2073`) and `styledCode` (`:1568-1579`) maps to
  fixed 16-colour names. Consolidate **before** adding surfaces that consume it.

### File diffs on edit (explicit user requirement)

**Krill has the rendering hooks but does not produce a diff.** What `edit_file`
actually returns (`EditTool.swift:44-50`) is:

```
Edited path/to/file.swift (+5 -2, 1 replacement(s)).
- <old string, newlines escaped to literal \n, clipped at 240 chars>
+ <new string, same>
```

`FileToolSupport.changeSummary` (`:45-51`) collapses each side to **one line**
with `\n` escapes and clips at 240 chars. So a multi-line edit renders as an
unreadable escaped blob, truncated. There are no line numbers, no context lines,
and no syntax highlighting.

The TUI half is already wired: `CodeView.swift:94-95` tags `+ `/`- ` prefixed
lines as `.diffAdd`/`.diffDel`, and `styledCode` (`ChatTUI.swift:1564-1565`) maps
them to green/red. But it's plain foreground colour with no row background tint,
and — critically — **tool results are collapsed by default**
(`toolOutputExpanded = false`), so even this much is hidden behind Ctrl-O; you
see `→ 3 lines · ctrl+o expand`. `/diff` exists (`ChatTUI.swift:1072-1084`) but
shells out to `git diff --stat` into a modal — a manual whole-repo view, not
per-edit inline.

**Plan:**

1. **A real line-based unified diff** in `FileToolSupport`, replacing
   `changeSummary`: hunks with N lines of context and true line numbers.
2. **Split the model-facing result from the UI-facing one** — this is where the
   `ToolResult.display` seam from Parts A/B pays off a third time. The model gets
   the compact `+5 -2` diffstat plus at most a small hunk (a full diff in the
   observation burns context on every edit and is a real cost at 128K on a 12B);
   the **full rendered diff rides `display`**, so the UI shows everything without
   spending a single model token. Add `ToolDisplay.diff(path:hunks:)`.
3. **Render it properly**: line-number gutter, `+`/`-` markers, and row
   *background* tints rather than foreground colour. Truecolor is already
   available (`Ansi` uses `38;2;r;g;b` freely) and the ember palette gives
   compatible tints.
4. **Expand diffs by default.** A diff the user must press Ctrl-O to see defeats
   the purpose. Diffs should be exempt from the global collapse — which is
   another reason to do the per-block collapse work rather than the single global
   Bool.
5. **Syntax highlighting is a separate, later step.** Nothing exists today —
   `TUIMarkdown` renders whole fences uniformly green and ignores the info string
   entirely. Ship the diff with gutters, context and tints first; per-language
   tinting is a genuinely large build and shouldn't gate the rest.

Also apply to `write_file` (`WriteTool.swift:46`) and `multi_edit`
(`MultiEditTool.swift:57`), which have the same shape.

**Two opencode sidebar rows have no Krill counterpart:** LSP status (no
integration exists anywhere) — drop it; and `$ spent` — Krill is a local runtime,
so the honest analogue is a `local · $0.00` badge or nothing.

---

## Where the rest lives

The ordered work plan (14 steps, with files and acceptance criteria per step) is
in **`docs/ASK_USER_IMPLEMENTATION.md`**. The full test matrix and manual QA
script are in **`docs/ASK_USER_VERIFICATION.md`**. The durable decision record,
including every rejected alternative and why, is
**`docs/decisions/0004-interactive-questions-and-adaptive-mode.md`**.

Deferred follow-ups this work surfaced but does not fix are tracked in
`docs/BACKLOG.md`.
