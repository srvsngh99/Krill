# Verification: ask_user, request_execute, adaptive mode, TUI lift

Companion to `docs/ASK_USER_AND_ADAPTIVE_MODE.md` (spec) and
`docs/ASK_USER_IMPLEMENTATION.md` (work order). This doc is how you prove the
work is correct.

Baseline commit: `973c885`.

## Ground rules

- **Test targets must never depend on `KrillCLI`, the executable target.** Its
  `@main` hijacks the swift-testing pass and breaks CI with a
  `--test-bundle-path` failure. Use `KrillHarnessTests`, `KrillTUITests`,
  `KrillServerTests`, `KrillEngineTests`. This is why `QuestionPrompt` lives in
  `KrillTUI` and `PermissionBox.chipLabel` lives in `KrillHarness` rather than
  inline in `ChatTUI`.
- **What is NOT unit-testable** (say so in the PR rather than faking coverage):
  `renderQuestion`'s ANSI output, the footer chip string as rendered,
  `handleAgentRunKey`. All are in the executable target. Keep the *logic* in the
  pure structs and QA the ANSI shell manually.
- **Build clean before judging a result.** Per repo history, stale artefacts have
  faked a verified fix here before: `--build-tests` is not `-c release`, and
  `--skip-build` judges the *last* build. A byte-identical result after a change
  is the tell. `swift package clean` when `KrillConfig` layout changes.

```sh
swift build && swift test
```

---

## 1. Unit tests

### `Tests/KrillHarnessTests/PermissionTests.swift` (extend)

| Assertion | Guards against |
| --- | --- |
| `parse("auto") == .acceptAll` | **The naming collision.** `"auto"` must keep meaning accept-all or every existing `config.toml` silently changes behaviour. |
| `Set(allCases.map(\.label)).count == 5` | **The test that would have caught the collision before it shipped.** Two modes sharing a label. |
| `Set(cycleOrder) == Set(allCases)` | A future mode added without a Shift+Tab entry. |
| `parse("adaptive"/"self"/"pilot") == .adaptive` | Synonym table. |
| `configuredDefault(nil)` and `configuredDefault("nonsense")` are `.plan` | Fail-closed default. |
| For **all five** modes: `decision("ask_user", isReadOnly: true) == .allow` | The "asking works in every posture" requirement, pinned across the matrix. |
| `.adaptive` denies a mutating tool and **never** returns `.ask` | A hard deny is the point — there is no human in the loop to prompt. |
| That deny reason names `request_execute` | The steer *is* the model's recovery signal. Pin the text. |
| `PermissionPolicy(mode: .executePosture).decision("bash", isReadOnly: false) == .ask` | **Promotion never hands over unprompted shell.** |

### `Tests/KrillHarnessTests/PermissionBoxTests.swift` (new)

| Assertion | Guards against |
| --- | --- |
| **`promote(to: .acceptAll)` returns `false` and changes nothing** | **The single most important safety assertion in this work.** An agent granting itself unattended shell. |
| `promote` from `.ask`/`.acceptEdits`/`.acceptAll` returns `false` | Promotion is monotonic and only from a planning posture. |
| A second `promote` returns `false` | Idempotence / anti-thrash. |
| `setPolicy(.plan)` re-arms a later `promote` | The human-demote path must not be wedged by a latch. |
| `origin` survives promotion | Identity vs posture — an adaptive run stays adaptive. |
| `chipLabel` flips `"adaptive (planning)"` → `"adaptive (executing)"` | The one user-visible requirement, made testable. |
| `deny: ["bash"]` survives promotion | `--deny-tool bash --permission-mode adaptive` must still never run bash. |
| Concurrent `promote` + `chipLabel` reads via `DispatchQueue.concurrentPerform` | Torn reads — the loop mutates on a background Task while the TUI renders on main. |

### `Tests/KrillHarnessTests/AskUserToolTests.swift` (new)

Reuse the existing `EchoTool: Tool` at `Tests/KrillHarnessTests/AgentEventTests.swift:17-24`
and copy `RecordingGate`'s actor shape from `PermissionTests.swift:39-44` for a
`StubAsker: UserQuestionGate`. (Do **not** copy `EchoTool` from
`Tests/KrillAgentTests/OperatorLoopTests.swift:31` — that conforms to
`OperatorTool`, a different protocol in an unrelated target.)

- `isReadOnly == true`; `parametersJSON` is valid JSON and
  `ToolCalling.argsSatisfySchema` accepts a well-formed call.
- Happy path: result contains `User answered: ...` and the numbered options.
- **Shape tolerance** — `options` as `[{label:…}]`, as `"a, b, c"`, as `[]`.
  These are the real failure modes: repair is presence-only so it will never fix
  a wrong-typed `options`.
- Missing `question` → `isError: true`.
- Declined → **`isError: false`** and the text contains the "state the
  assumption" wording. This is the anti-retry-loop contract.
- Timeout: an asker that never resolves + `timeout: 0.05` returns unanswered
  **and** `cancelPending()` was recorded — proving no leaked continuation.

### `Tests/KrillHarnessTests/RequestExecuteToolTests.swift` (new)

- `isReadOnly == true`, **and** `PermissionPolicy(mode: .plan)
  .decision("request_execute", isReadOnly: true) == .allow` — pins the reductio
  that a mutating version would be denied by the very mode it exits.
- `origin == .plan`, option 1 → box is `.acceptEdits`, `isError == false`,
  content names the new posture **and contains the "ignore the earlier PLAN MODE
  instruction" sentence**.
- Option 2 → `.ask`. Option 3 / declined / timeout → box **unchanged**,
  `isError == false`, and the content does **not** contain "call
  request_execute again".
- `origin == .adaptive` → **the stub asker records ZERO questions** and the box
  is `.acceptEdits`. Self-promotion must not prompt.
- Called twice → second returns `isError: false` "already implementing", box
  unchanged.
- Posture already `.acceptEdits` → no-op, no question raised.
- Missing `summary` → does not hard-fail (schema strictness exists to drive
  repair, not to reject at execution).

### `Tests/KrillHarnessTests/AgentLoopTests.swift` (extend)

**The two most valuable tests in the whole change** — only these prove the
decision site actually re-reads the box mid-run.

1. Scripted generator emits `request_execute` then `write_file` then a final
   answer; loop built with `PermissionPolicy(mode: .plan)` + a `PermissionBox` +
   an approving stub asker. Assert `write_file`'s observation is **not**
   "Permission denied", and that `allow`/`deny` survived the flip.
2. Same, with a declining asker → assert it **is** denied.
3. Adaptive variant: `write_file` (denied — assert the observation names
   `request_execute`) → `request_execute` → `write_file` (allowed).
4. `ask_user` round-trip: assert `transcript.messages` contains a `user`-role
   turn `Tool result (ask_user):` carrying the answer, and that the loop reached
   `.finalAnswer` — i.e. the answer reached the model and the run continued
   rather than terminating.

### Other

- `Tests/KrillHarnessTests/AgentEntryTests.swift` — `foldAgentEvent` on the new
  event yields exactly one `.note`.
- `Tests/KrillHarnessTests/ToolNameAliasConformanceTests.swift` — add both tools
  to `allRegisteredToolNames()` (`:22-28`); the existing
  `testEveryAliasTargetIsARegisteredTool` then pins all the new aliases
  automatically. **This test will fail until you do**, which is the intended
  forcing function.
- **Prompt test** (new, `KrillHarnessTests`) — build the system prompt for all
  five postures; assert `askUserDirective` is present in **every** one, and
  `planSystemSteer` present exactly when `mode.initialEffective == .plan`.
- `Tests/KrillTUITests/QuestionPromptTests.swift` (new) — selection wraps both
  directions; single option; empty options (free-text row only);
  `index(forDigit:)` in and out of range; the free-text row is always last.
- `Tests/KrillServerTests/AgentSessionTests.swift` — clone the three
  `RemoteApprover` tests using its `waitForPending` helper (`:14-26`):
  park/resolve, **stale-id rejection leaves the question pending** (mirrors
  `:44-55`), `cancelPending` resumes as declined. Plus: a box flip is visible
  through `summary()`.
- `Tests/KrillServerTests/AgentAPITests.swift` —
  `testPostQuestionWithNothingPendingReturns409`, cloned from the approvals
  equivalent at `:198`.
- `Tests/KrillEngineTests/` — **the stats race regression**: a consumer that
  breaks on the first `isEnd` and immediately reads `stats()` observes non-nil.
  Without an intervening `await`, matching `EngineGenerator.collect`'s shape.

---

## 2. Manual QA

Use the `krill-qa` skill. Build and install a dev binary first — note `brew`'s
`krill` shadows `/usr/local/bin`, so `brew unlink krill` before installing.

### TUI (`krill`)

1. **Ask, in plan mode.** Agent surface, plan posture. Give a deliberately
   ambiguous task ("add caching"). → widget appears with options; answer it;
   **the run continues** rather than ending the turn.
2. **Ask, in every other posture.** Repeat in ask / accept-edits / auto. The
   question must appear in all of them — that is the "building it right matters
   more" requirement.
3. **Approve a plan.** In plan mode, let it finish a plan → `request_execute`
   prompts with three options → pick option 1 → chip flips to `accept-edits`,
   and a subsequent edit **actually applies**.
4. **Decline a plan.** Pick "keep planning" → posture unchanged, and the agent
   does **not** immediately re-ask.
5. **Adaptive.** Shift+Tab to adaptive → give a task → it self-promotes with
   **no prompt**, the chip reads `adaptive (planning)` → `adaptive (executing)`,
   a note appears in the transcript, and **the first `bash` still prompts**.
6. **Re-leash mid-run.** While a turn is running, press Shift+Tab → posture
   tightens. (This is inert today; step 10.6 adds it.)
7. **Back-to-back questions.** A turn that calls `ask_user` then
   `request_execute` — the second widget must not render the first's content.
8. **Background agents.** `/bg` a task, then `/agents` — each session has its own
   question, its own posture, and its own todo list. A child promoting itself
   must **not** change the foreground posture.

### CLI (`krill code`)

9. `krill code --plan "<task>"` on a tty → prompts, promotes, edits apply, first
   `bash` prompts.
10. `krill code --plan "<task>" </dev/null` → **must not hang** and must not
    offer either tool.
11. `krill code --permission-mode adaptive "<task>"` → self-promotes; the first
    `bash` prompts (this is the `gate: nil` deadlock fix — before step 9.2 it
    would be silently denied with no prompt printed).
12. `krill code --deny-tool bash --permission-mode adaptive "<task>"` → bash
    stays denied after promotion.

### Remote / phone (`krill ui`)

13. Same ask and approve flows through the question card; the mode pill updates
    live.
14. **Five-button picker at phone width** — confirm the `.seg` row does not wrap
    badly; fall back to a `<select>` if it does.
15. **Reconnect mid-question.** Close the tab while a question is parked, reopen
    → the pending question is rendered from `summary()`, not lost.
16. **Abandon a session.** Start an adaptive session, close the tab, let it
    promote and hit a `bash` → it must time out and deny after 300s rather than
    hanging forever.

### Context and diffs

17. Footer shows live `ctx <bar> N/128K P%` throughout a multi-turn agent
    session, and `/context` no longer says "estimated".
18. A multi-line edit renders as a readable diff — line numbers, context, tinted
    rows — **without pressing Ctrl-O**. Then check the *transcript*, not just the
    screen: the model's observation must have stayed a compact diffstat.
19. Sidebar shows the task list with the active item marked, `N/M done`, and
    live session token totals. Resize below 110 cols → sidebar disappears
    cleanly and `/context` still works.

---

## 3. Regression checklist

Things that are easy to break and cheap to check:

- `krill run`, `krill serve`, and the OpenAI/Anthropic/Ollama compat paths are
  untouched by any of this. Smoke one request through each.
- A model with **no** tool support still works — `ToolRegistry.specs()` empty
  means neither new tool is advertised.
- Existing `config.toml` values (`plan`, `ask`, `accept-edits`, `auto`) all still
  parse to what they parsed to before. Especially `auto`.
- The prompt bytes for a non-plan, non-adaptive posture are unchanged by step 2's
  hoist.
- `krill launch <agent>` and the tap/formula flows are unaffected.
- **Public build has no Kreach strings** — run the usual release grep before
  tagging; unrelated to this work but gated on every release.
