# Implementation plan: ask_user, request_execute, adaptive mode, TUI lift

Companion to **`docs/ASK_USER_AND_ADAPTIVE_MODE.md`** (the spec — read it first
for *why*). This doc is the ordered *how*. Verification lives in
`docs/ASK_USER_VERIFICATION.md`.

Baseline commit: `973c885`. All line numbers are from that commit — re-check
them if the branch has moved.

## How to use this doc

Steps are ordered by dependency, not by size. Each has **Files**, **Do**, and
**Done when** — treat "Done when" as the gate before moving on. Steps 1 and 2 are
prerequisites for almost everything; do not reorder them.

Suggested PR split (each is independently shippable and reviewable):

| PR | Steps | Rationale |
| --- | --- | --- |
| 1 | 1 | Pure bug fixes, no new surface. Ships value on its own. |
| 2 | 2 | Pure refactor, no behaviour change. Keeps the diff of PR 3 readable. |
| 3 | 3–8 | The harness core: seam, both tools, the posture model, prompts. |
| 4 | 9–12 | The three surfaces + web UI. |
| 5 | 13 | The TUI lift (can itself be split — see step 13). |
| 6 | 14 | Docs. |

---

## Step 1 — Fix the stats race and four related defects

**Why first:** the Part C sidebar renders context usage. Build it before this and
it displays a 4-chars/token *estimate* presented as measured truth.

**Files**
- `Sources/KrillEngine/InferenceEngine.swift`
- `Sources/KrillCLI/TUI/ChatTUI.swift`
- `Sources/KrillHarness/HarnessGenerator.swift`
- `Sources/KrillCLI/Code/CodeCommand.swift`, `Sources/KrillCLI/TUI/AgentSession.swift`

**Do**
1. **The root cause.** In `InferenceEngine.generate(messages:)`, move the
   `statsHolder.stats` write (`:2021`) to *before* every terminal `isEnd` yield
   (`:1787, :1805, :1863, :1893, :1929, :1955, :1998`). Match the ordering the
   five VLM drivers already use and document at `:2269-2281`. Copy that comment's
   reasoning to the dense path so the next person does not re-invert it.
2. **Final drain.** `ChatTUI.swift:1380` is the only `agentStats.take()`; the
   loop breaks at `:1392` and `:1399-1404` never re-checks. Add a final `take()`
   after the loop so a `put` landing between `:1380` and `:1382` is not lost.
3. **Dead logit mask.** Promote
   `complete(messages:constrainingToolNames:)` from the protocol *extension*
   (`HarnessGenerator.swift:45-49`) to a protocol *requirement* (beside `:27`),
   keeping the extension body as the default. Today the call at
   `AgentLoop.swift:199` on `any HarnessGenerator` statically dispatches to the
   default, so `EngineGenerator.swift:43-63` — "layer 0", the primary defence
   against a model naming an unoffered tool — has never run in production.
4. **Unwired `onStats`.** Set it at the three sites that construct
   `EngineGenerator` without it: `CodeCommand.swift:215`,
   `AgentSession.swift:56`, `ChatTUI.swift:1422`.
5. **Routing order.** `processSubmit` routes to `activeSession`
   (`ChatTUI.swift:821-825`) *before* the `surface == .agent` check (`:838`), so
   an attached background session always uses the no-`onStats` generator. Move
   the check.

**Done when** the footer shows `ctx <bar> N/128K P%` after a normal agent turn,
`/context` no longer says "estimated", and the new `InferenceEngine` regression
test passes.

---

## Step 2 — Hoist the plan-mode steer (prerequisite refactor, no behaviour change)

**Why:** the steer has **four** verbatim copies. Steps 5 and 8 must change it,
and adaptive mode adds a `== .plan` predicate bug at each site (see spec).
Hoisting means the predicate changes once.

**Files** `Sources/KrillHarness/AgentEnvironment.swift`,
`Sources/KrillCLI/Code/CodeCommand.swift:155-160`,
`Sources/KrillServer/AgentAPI.swift:224-229`,
`Sources/KrillCLI/TUI/ChatTUI.swift:1349-1352`,
`Sources/KrillCLI/TUI/AgentSession.swift:49-52`

**Do** Add `AgentEnvironment.planSystemSteer` and
`AgentEnvironment.planTurnPrefix`; replace all four literals with references.
While here, fix the stale doc comment at `AgentEnvironment.swift:10-11` claiming
`toolDirective` mirrors `ToolCalling.agenticToolDirective` — they diverged (the
former carries an extra training-cutoff paragraph). Say they deliberately differ
and why.

**Done when** the four sites reference the statics, the emitted prompt bytes are
byte-identical to before, and existing tests pass unchanged.

---

## Step 3 — The question seam and both `ToolResult` fields

**Files** `Sources/KrillHarness/UserQuestion.swift` (new),
`Sources/KrillHarness/Tool.swift:5-13`

**Do**
1. New file with `UserQuestion {header, question, body, options}`,
   `UserAnswer {text, optionIndex, wasFreeText, declined}`, and
   `protocol UserQuestionGate { func ask(_:) async -> UserAnswer;
   func cancelPending() }`. `optionIndex` and `body` are required by step 7 —
   add them now, not as a retrofit.
2. **One edit** to `ToolResult` adding **both** optional fields, each defaulted
   `nil` so all 14 tools and every call site compile unchanged:
   - `display: ToolDisplay?` — `.question(...)`, later `.diff(...)` (step 13)
   - `effect: ToolEffect?` — `.permissionMode(PermissionMode)`

**Done when** the package builds with no changes to any existing tool.

---

## Step 4 — `AskUserTool`

**Files** `Sources/KrillHarness/Tools/AskUserTool.swift` (new)

**Do** Implement per the spec: flat one-question schema, `options` as
`[String]`, `isReadOnly = true`, gate injected via `init(gate:timeout:)`.

Three things that are easy to get wrong:
- **Parse defensively.** `argsSatisfySchema` is presence-only and the repair pass
  is capped at 256 tokens, so repair will *not* fix a wrong-typed `options`.
  Accept `[String]`, `[{label|text|title|value}]`, and a single delimited string;
  fall back `question` → `prompt`/`text`; dedupe; cap at 6.
- **Never abandon the continuation.** Race `gate.ask` against `Task.sleep` in a
  `withTaskGroup`; when the sleeper wins, call `gate.cancelPending()` so the
  losing child returns and the group drains. Wrap in
  `withTaskCancellationHandler { gate.cancelPending() }`.
- **Declined is `isError: false`** and its text says *"state the assumption you
  made, and do not ask again."* An error result pushes the model into a retry
  loop.

**Done when** `AskUserToolTests` passes (see verification doc).

---

## Step 5 — `PermissionMode.adaptive`

**Files** `Sources/KrillHarness/Permission.swift`

**Do**
1. `case adaptive` (rawValue `"adaptive"`); `parse` arm for
   `"adaptive"/"self"/"pilot"`. **Leave `"auto"` → `.acceptAll` untouched** —
   re-pointing it would silently reinterpret every existing `config.toml`.
2. `label` → `"adaptive"`; `summary` per spec; `cycleOrder` →
   `[.plan, .adaptive, .ask, .acceptEdits, .acceptAll]`.
3. `initialEffective` (`.adaptive → .plan`, else self) and
   `static let executePosture: PermissionMode = .acceptEdits`.
4. A **fail-closed** `.adaptive` arm in `decision`'s switch sharing `.plan`'s
   deny — reaching it means a policy was built from a non-effective posture,
   which is a wiring bug and must fail read-only. Its reason string should name
   `request_execute`: the steer *is* the recovery signal.
5. `configuredDefault` unchanged (still falls back to `.plan`).

**Watch out:** `CodeCommand.swift:195-206` is an exhaustive switch and **will
fail the build** — that is the forcing function. Add the `.adaptive` arm.
But every `== .plan` comparison compiles fine and silently skips the plan steer
for adaptive; step 2's hoist is what lets you fix that in one place, by changing
the predicate to `mode.initialEffective == .plan`.

**Done when** `PermissionTests` passes including the `"auto"` regression pin.

---

## Step 6 — `PermissionBox` and the loop read

**Files** `Sources/KrillHarness/PermissionBox.swift` (new),
`Sources/KrillHarness/AgentLoop.swift`, `Sources/KrillHarness/AgentEntry.swift`

**Do**
1. New `PermissionBox` per the spec: `origin` (immutable identity), locked
   `policy`, `effective`, `isPlanning`, `chipLabel`, plus the two mutators —
   `promote(to:)` (model-facing, guarded) and `setPolicy(mode:)` (human-facing,
   unguarded). **`promote` must refuse anything but `.ask`/`.acceptEdits`, and
   refuse to act unless currently `.plan`/`.adaptive`.** That guard is the
   structural reason an agent can never grant itself `.acceptAll`.
   `promote` rebuilds the policy carrying `allow`/`deny` forward.
2. `AgentLoop` gains `public let permissionBox: PermissionBox?` defaulted `nil`;
   the decision site (`:272-273`) reads
   `(permissionBox?.policy ?? permission).decision(...)` **inside** the per-call
   loop — that placement is what makes a flip effective on the next tool call in
   the same turn.
3. `AgentEvent.permissionChanged(from:to:)`, emitted from `record(...)`
   (`:243-259`) when `result.effect` is `.permissionMode`. The loop relays a
   declared effect; it must never learn a tool name.
4. `foldAgentEvent` maps it to the **existing** `AgentEntry.note` — no new entry
   case, so all four consumers render it for free.

**Done when** `PermissionBoxTests` passes and existing `AgentLoopTests` compile
untouched (the `nil` default proves the change is additive).

---

## Step 7 — `RequestExecuteTool`

**Files** `Sources/KrillHarness/Tools/RequestExecuteTool.swift` (new),
`Sources/KrillTooling/ToolCalling.swift:1394-1413`

**Do** Implement the three posture arms (ask / self-promote / no-op) per the
spec. Register aliases `exitplanmode`, `plan_exit`, `exitplan`, `approve_plan`,
`start_implementing`. Reuse `UserQuestionGate` for the approval prompt — do not
add a second channel.

Two easy mistakes:
- The "already implementing" branch must test the **current effective mode**, not
  a `hasPromoted` latch. If a human demotes mid-run, the agent may legitimately
  ask again; a latch wedges that.
- Decline is `isError: false` and must **not** say "call it again when ready" —
  the runaway guard keys on `name + argumentsJSON`, so a reworded `summary`
  sails straight through and re-prompts.

**Done when** `RequestExecuteToolTests` passes, including the assertion that the
adaptive path invokes the gate **zero** times.

---

## Step 8 — Prompts

**Files** `Sources/KrillHarness/AgentEnvironment.swift` plus the four builders
from step 2.

**Do**
1. Amend `toolDirective` (`:11-16`): its *"Once you have the tool results, reply
   with the final answer"* literally tells the model to stop after an answer. Add
   the `ask_user` exception.
2. New `askUserDirective`, appended **unconditionally at all four builders**.
   ⚠️ At three of them the append site sits immediately adjacent to an
   `if mode == .plan` block (`CodeCommand.swift:154`, `AgentAPI.swift:223`,
   `ChatTUI.swift:1350`) — it must **never** land inside one.
3. Extend `planSystemSteer`: `request_execute` is how you leave plan mode, and
   `ask_user` is permitted here.
4. Add the adaptive tail to the steer (selected on
   `mode.initialEffective == .plan`, with adaptive-specific wording).

**Done when** the prompt test passes: `askUserDirective` present for **all five**
postures, `planSystemSteer` present exactly when `initialEffective == .plan`.

> Do this before any manual model testing. Without it the model never calls
> either tool and you will conclude, wrongly, that the tools are broken.

---

## Step 9 — CLI surface

**Files** `Sources/KrillCLI/Code/StdinQuestionAsker.swift` (new),
`Sources/KrillCLI/Code/CodeCommand.swift`

**Do**
1. `StdinQuestionAsker` mirroring `StdinApprover`, including its EOF discipline
   (`:26-30`, `:45-53`): numbered options, `readLine()` bridged off the
   cooperative pool, blank/EOF → declined.
2. **The gate fix** at `:220`:
   `gate: (mode != .acceptAll && RawTerminal.isInteractive) ? StdinApprover() : nil`.
   Today `nil` in plan and adaptive means that after a promotion every `bash`
   call is silently denied forever, with no prompt printed.
3. Register `ask_user` only when `RawTerminal.isInteractive`. Register
   `request_execute` when interactive **or when `mode == .adaptive`**: adaptive
   self-promotion does not consult a human gate and must work in scripts/CI.
   Supply a `DecliningQuestionGate` for the non-interactive adaptive case so any
   accidental planning-path question fails safe without hanging. A piped `.plan`
   run offers neither tool.

**Done when** `krill code --plan "<task>"` on a tty prompts and promotes; the same
command with `</dev/null` neither hangs nor offers the tools; and
`krill code --permission-mode adaptive "<task>" </dev/null` offers
`request_execute` and can self-promote to `.acceptEdits`.

---

## Step 10 — TUI surface

**Files** `Sources/KrillTUI/QuestionPrompt.swift` (new),
`Sources/KrillCLI/TUI/TUIQuestionAsker.swift` (new),
`Sources/KrillCLI/TUI/ChatTUI.swift`, `Sources/KrillCLI/TUI/AgentSession.swift`

**Do**
1. `QuestionPrompt` — pure, no ANSI, in `KrillTUI` so it is testable (tests must
   not depend on the `KrillCLI` executable target). Model it on `AgentPicker`.
2. `TUIQuestionAsker` — copy `TUIApprover`'s structure verbatim (`NSLock`, sync
   `register` helper, `@unchecked Sendable`). No `sticky` set; "always allow" has
   no analogue for a question.
3. `ChatTUI`: `agentTools(asker:)`; build the widget from `asker.pending()` on
   the render tick — **not** from an `AgentEvent`, which structurally cannot
   arrive in time. **Cache it keyed on the `UserQuestion` value**, not on
   nil→non-nil, or back-to-back questions in one turn render the stale widget.
4. `renderQuestion` beside `renderPicker` (`:763`); key branch in
   `handleAgentRunKey` (`:1505`) **before** the approver branch.
5. Footer chip → `box.chipLabel` (`:1787-1788`, and `:1782` for attached
   sessions).
6. **Add a `.backTab` arm to `handleAgentRunKey`** — Shift+Tab is inert mid-turn
   today, so there is currently no way to re-leash a running agent.
7. Per-session `asker`, `PermissionBox`, and `TodoTool` on `AgentSession`
   (`spawnSession`, `:1591-1596`) — build the session first, then its registry.

**Done when** the manual TUI flows in the verification doc pass.

> If free-text entry proves fiddly, ship options + Esc-to-skip first. The tool
> result already distinguishes the two cases.

---

## Step 11 — Remote surface

**Files** `Sources/KrillServer/AgentSessions.swift`,
`Sources/KrillServer/AgentAPI.swift`

**Do**
1. `RemoteQuestionAsker` mirroring `RemoteApprover` (`:74-135`) **including its
   stale-id protection** (`:126`).
2. `POST /v1/agent/sessions/{id}/questions` — clone the approvals handler
   (`AgentAPI.swift:60-73`) byte-for-byte, including the 409 on nothing pending.
   Add it to the route-table doc comment at `:9-19`.
3. SSE frames `question_request`, `question_answered`, `permission_changed`.
4. `RemoteAgentSession`: keep `mode` as immutable *origin*, add the box; extend
   `summary()` (`:212-222`) with the effective posture and phase so a
   reconnecting client renders correctly without SSE replay.
5. Register both tools in `agentToolRegistry(for:)` (`AgentAPI.swift:209-215`) —
   this is a hand-listed subset, so it is not automatic.
6. **Separate commit:** give `RemoteApprover.approve` (`:88-92`) a 300s timeout
   resolving to **deny**. A remote session whose client closed currently parks on
   a continuation forever. Pre-existing, but adaptive makes it reachable.

**Done when** `AgentSessionTests` and `AgentAPITests` pass, including the 409.

---

## Step 12 — Web UI

**Files** `Sources/KrillServer/WebUI.swift`

**Do** `addQuestion`/`resolveQuestion` modelled on `addApproval` (`:695-721`),
rendering `body` in a `<pre>` plus one button per option, a free-text input, and
Skip. Two new `applyEvent` cases (`:740-741`). A `permission_changed` handler
updating `cur.meta` and the `#s-sub` subtitle (`:799-800`). Fifth segment button
in the picker (`:457-462`).

**Done when** the phone flows pass — and **check the `.seg` row does not wrap
badly at five buttons on a phone width**; fall back to a `<select>` if it does.

---

## Step 13 — The TUI lift

Independently shippable sub-parts; ship in this order.

1. **`TodoTool.snapshot()`** — lock-guarded public accessor, plus the
   per-session instance fix from step 10.7. Infer `active` = first not-done item
   rather than adding a `status` field to the schema.
2. **Layout primitives** — `positioned(row:col:)`, a `join(left:right:width:)`
   row composer, and a public visible-width clip in `KrillTUI`. Thread
   `mainWidth = cols - sidebarWidth - 1` through `paneLines`,
   `TUIMarkdown.render`, `CodeView.*`, `renderMenu`, `inputBox`. All pane content
   already re-wraps from source every frame, so there is no cache to invalidate.
3. **The sidebar** — ~34 cols, only when `cols >= 110`, toggleable. Move the
   `/context` panel (`:1197-1322`) into it; add the task list and the session
   totals that today are only printed after the TUI exits (`:329-353`).
4. **Real diffs** — a line-based unified diff in `FileToolSupport` replacing
   `changeSummary` (`:45-51`, which escapes newlines and clips at 240 chars).
   Model-facing result stays a compact diffstat; the **full diff rides
   `ToolResult.display`** so the UI shows it without spending model context. Add
   `ToolDisplay.diff(path:hunks:)`. Render with a line-number gutter and row
   *background* tints. **Exempt diffs from the global collapse** — a diff you
   must press Ctrl-O to see defeats the purpose. Apply to `write_file` and
   `multi_edit` too.
5. **Polish, each independent** — per-block collapse (SGR mouse is already
   enabled and clicks already parsed-then-discarded, so click-to-expand is
   reachable); the `+ Thought: 1.5s` reasoning block (net-new — the filter
   currently discards think tokens with no accessor); markdown lists/tables/links
   (lists are currently reflowed into paragraphs, destroying structure); palette
   consolidation (the ember stops are triplicated).

**Done when** the sidebar shows live task progress and context, and a multi-line
edit renders as a readable diff without pressing Ctrl-O.

---

## Step 14 — Docs

Update `docs/TUI.md:40,:105-106,:121` and `docs/AGENT_UI.md:121,:132` (both
enumerate the four modes verbatim), `docs/SERVER_API.md` (the new route and
frames), and `CHANGELOG.md`. Mark
`docs/decisions/0004-interactive-questions-and-adaptive-mode.md` adopted only after the manual QA gate passes (keep it proposed during review).

**CHANGELOG must note the forward-compat behaviour:** an older Krill reading
`default_agent_permissions = "adaptive"` gets `parse` → nil →
`configuredDefault` → `.plan`. Incompatible in the safe direction.
