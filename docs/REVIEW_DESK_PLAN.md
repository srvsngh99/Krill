# Review Desk — opt-in hunk review + comment-driven fixes in `krill code`

Status: DRAFT under ideation (2026-07-18).

Decided so far (2026-07-18, with Sourav):
- **Opt-in, not a gate.** Review is an on-demand surface (`/review`, ctrl+r,
  `krill diff`), used when the user chooses — typically before raising a PR,
  to manually correct something, or to leave inline comments. There is NO
  forced turn-end review and NO new permission posture. (An earlier draft
  had a blocking `review` posture; explicitly rejected.)
- **Apply-then-review**: agent edits hit disk as they do today; review works
  over the resulting changeset. No staging overlay — the agent must be able
  to build/test its own changes.
- **Git-backed changesets are the primary source** (see Review model). This
  also captures bash/formatter/codegen changes, closing most of the
  edit-tool-only gap.
- **Inline comments → agent addresses them** is the flagship loop (see
  Comment loop). Review is not just inspection; it authors the next round of
  agent work.
- **Boring names**: `/review`, `krill diff`, `krill show`. No sub-brand.

Locked 2026-07-19 (Sourav):
- **Comments are repo-scoped**, persisted under `.krill/` (gitignored) so
  they survive across sessions until resolved.
- **Default diff base = HEAD** (quick spot-check view); one key (`b`) flips
  to merge-base-with-main (pre-PR view); last choice remembered per repo.
- **Agent cannot self-resolve comments**: when its addressing turn completes
  and the anchor text changed, a comment becomes "addressed, unconfirmed";
  only the user confirms (or re-opens) in `/review`.
- Untracked/new files render collapsed (`new file, N lines`), expand on
  enter.

## Why

Local models make more mistakes than frontier cloud models, so the human
pass over a changeset matters more, not less. Today that pass happens in
`git diff` plain text or an external tool that knows nothing about the
agent. The Review Desk makes the pass native: a hunk-by-hunk review UI where
every action can flow *back into the model* — a comment on a hunk becomes a
work item the agent addresses. Standalone viewers (hunk/hunkdiff) render
diffs; they cannot talk back to the model. We own the loop, so review
becomes steering.

Phase 1 of bringing hunk + herdr's ideas native. Decided 2026-07-19: there
will be **NO session multiplexer** (no panes, no PTY hosting, no herd
product). herdr's value is instead *extracted* into the existing
single-session `krill code` — see "herdr extractions" at the end of this doc.

## Product shape (v1)

One engine, three surfaces:

1. **`/review` (or ctrl+r) in-session** — full-screen review of the current
   changeset: file list on the left, hunks on the right, keyboard-driven.
   Per hunk: **[c]omment** (the flagship action), **[x] revert**,
   **[e] open in $EDITOR at this line** (manual correction), **[k]eep /
   next**. On exit, queued comments can be handed to the agent in one
   keystroke.
2. **Turn diffstat line** — after any turn that edited files:
   `~ 3 files changed, +42 -17  (/review to inspect)`. Discoverability hook;
   zero cost when ignored.
3. **`krill diff` / `krill show`** — the same renderer as a standalone
   command (working tree, a commit, or `--base <branch>` for the whole PR
   diff). Useful even on changes made by other tools. Watch mode deferred.

### Review model: git-backed changeset, snapshot fallback

- **In a git repo (the normal case)**: the changeset is `git diff` output —
  working tree vs HEAD by default, or vs `--base <branch>` for pre-PR
  review of the whole branch. Git supplies pre-images, rename/untracked
  handling, and captures *every* change regardless of which tool made it
  (edit tools, bash, formatters). Revert = apply the hunk's reverse patch.
- **Outside git**: fall back to the harness `TurnSnapshot` layer (pre-image
  recorded the first time a mutating tool touches a file in the session).
  Covers edit-tool changes only; documented limitation.

### The comment loop (the moat)

Comments are the point. The flow mirrors a PR review, but local and instant:

1. User walks the diff, hits `c` on a hunk, types a line
   ("this cache is never invalidated", "rename this, unclear").
2. Comments accumulate in a session-scoped store, each anchored to
   file + line range + the hunk text at comment time.
3. On leaving the screen (or via `/address`), the harness offers:
   `3 comments queued — send to agent? [y/N]`. Yes → the agent's next turn
   opens with a structured, machine-labeled block: each comment with its
   file:line anchor and the hunk excerpt, plus the instruction to address
   each and report per-item.
4. The agent works through them; addressed comments are marked resolved in
   the store. Re-opening `/review` shows unresolved comments still pinned
   to their (re-anchored) locations.

Revert also feeds back, quietly: a reverted hunk queues a note ("user
reverted this change: <hunk>") so the model doesn't rediscover and redo it.
Keep is silent. Manual `$EDITOR` fixes need no note — the next changeset
simply reflects them (the model re-reads files anyway).

## UX sketches

Transcript hook after a turn:

```
  ▸ edit_file  Sources/KrillEngine/Decoder.swift
      Edited Sources/KrillEngine/Decoder.swift (+9 -3, 1 replacement).
  Done. The decoder now caches the rotary embeddings between steps.

  ~ 2 files changed, +14 -5   /review to inspect
```

Review screen (full-screen takeover, same overlay pattern as AgentPicker):

```
 REVIEW  working tree vs main — 2 files  +14 -5      [3/5 hunks]  2 comments
┌──────────────────────────┬───────────────────────────────────────────────┐
│ Sources/KrillEngine/     │ Decoder.swift  ── hunk 3/4  @@ -118,7 +118,9 │
│  ▸ Decoder.swift  +9 -3  │                                               │
│    Sampler.swift  +5 -2  │   118    func step(_ token: Int32) {          │
│                          │ - 119        let rot = makeRotary(pos)        │
│                          │ + 119        let rot = rotaryCache[pos]       │
│                          │ + 120            ?? makeRotary(pos)           │
│                          │   121        ...                              │
│                          │                                               │
│                          │ ✦ agent: reuse cached rotary; rebuilt only on │
│                          │   cache miss                                  │
│                          │ ● you: cache is never invalidated on reset —  │
│                          │   clear it in resetState()                    │
├──────────────────────────┴───────────────────────────────────────────────┤
│ [c]omment  [x] revert  [e] editor  [n/p] hunk  [tab] file  [q] done      │
└──────────────────────────────────────────────────────────────────────────┘
```

On quit with queued comments:

```
  2 comments queued — send to agent now? [y/N]
  > y
  ▸ addressing review comments (2)
    1. Decoder.swift:119 — cache never invalidated on reset … working
```

`ask`-mode gate prompt, upgraded (kept from earlier draft — small,
orthogonal, still worth doing):

```
  edit_file → Sources/KrillEngine/Decoder.swift   +9 -3

    @@ -118,7 +118,9 @@
      func step(_ token: Int32) {
  -       let rot = makeRotary(pos)
  +       let rot = rotaryCache[pos] ?? makeRotary(pos)

  Allow? [y]es / [N]o / [a]lways / [r]eason…
```

## Architecture

New pure module **KrillDiff** (no deps, unit-testable — same philosophy as
CodeView/Layout):

- Unified-diff parser (`git diff` output → `[FileChange]` → `[Hunk]`).
- Myers line diff (for the snapshot fallback and gate-prompt rendering,
  where there is no git output to parse).
- Reverse-patch generation for hunk revert; re-anchoring of comment
  positions after the file changes (context-line matching, best-effort).
- Diffstat.

Changeset providers:

- `GitChangeset` (KrillTooling or KrillHarness): shells out to
  `git diff --no-color` / `git show`; parses via KrillDiff. Handles
  `--base`, untracked files (`--intent-to-add` trick or manual /dev/null
  diff).
- `TurnSnapshot` actor (KrillHarness): pre-image on first mutation per file
  per session, hooked at the single `isFileEdit` dispatch choke point in
  `AgentLoop`. Powers the diffstat line and the non-git fallback.

Comment store:

- `ReviewComment { file, lineRange, anchorText, body, state (open/resolved),
  createdTurn }`, session-scoped, persisted under the session state dir so a
  crash doesn't eat comments. Not committed to the repo.
- Delivery: a machine-labeled block prepended to the next user turn
  (`[review comments] 1. path:line … 2. …` + per-item address instruction).
  Resolution: v1 marks a comment resolved when the agent's addressing turn
  completes and the anchor text changed; `/review` lets the user re-open.

TUI additions (KrillTUI):

- `DiffView`: pure formatter `[FileChange] -> [CodeLine]`, reusing
  `.diffAdd/.diffDel` + new roles (`.hunkHeader`, `.filePath`,
  `.annotation`, `.comment`). Unified layout only in v1.
- `ReviewScreen`: full-screen interactive state machine (selection, per-hunk
  status, comment input line), same overlay pattern as AgentPicker/SlashMenu.
- `[e] editor`: spawn `$EDITOR +<line> <file>` suspending the TUI (same
  terminal discipline as external command execution), refresh changeset on
  return.

Edit tools: optional `why` string param (edit/write/multi_edit) rendered as
the `✦ agent:` annotation per hunk. Optional → older prompts keep working.

`PermissionGate` gains a default-implemented richer method for the upgraded
ask prompt: `approve(edit: ProposedEdit) -> GateAnswer`
(allow / deny(reason:) / always), with `ProposedEdit` carrying path + hunks.

## What we are NOT building (v1)

- Any blocking/forced review mode or new permission posture — rejected.
- Side-by-side layout, mouse, pager mode, git difftool integration.
- Watch mode — v1.1.
- In-TUI text editing of hunks (that's what `[e] editor` is for).
- Committing/PR-raising from the review screen — the user's existing git
  flow owns that; we only prepare the changeset. (Possible v2: `[P] raise
  PR` handing off to `gh`.)
- Any session multiplexer / pane / PTY surface — rejected outright.
- Syntax highlighting inside hunks — revisit post-v1 (KrillGrammar may make
  it nearly free).

## Sequencing (one PR each, all additive)

1. **KrillDiff** module + tests (parser, Myers, reverse patch, diffstat).
2. **`krill diff` / `krill show`** standalone read-only viewer over
   GitChangeset — proves the renderer with zero harness coupling.
3. **`/review` in-session** — same screen inside `krill code`, plus the
   turn diffstat hook (TurnSnapshot) — still read-only navigation.
4. **Comment loop** — `[c]`, store, `/address` handoff, resolution
   tracking. The flagship lands here.
5. **Hunk actions** — `[x] revert` (reverse patch + feedback note),
   `[e] editor` jump.
6. **Upgraded `ask` gate** — diff-rendered approval + deny-with-reason.
7. v1.1: watch mode, snapshot-fallback polish, `why` annotation nudges,
   syntax highlight.

## Open questions (ideation)

1. **Comment delivery voice.** Machine-labeled block vs plain user-voice
   text ("address these: …"). Local models follow user-voice better, but a
   labeled block keeps the user's words distinguishable from harness prose.
   Needs a quick empirical check on gemma/qwen before PR 4 lands.

(Former questions on persistence scope, default base, address verification,
and untracked-file rendering are resolved — see the locked decisions at the
top.)

## herdr extractions (decided: no multiplexer — steal the ideas, not the panes)

`krill code` already runs multiple agents in one process: `dispatch_agent`
spawns independent background `AgentSession`s and the `/agents` switcher
attaches to them. That is the herd — it just lacks herdr's *awareness*.
Extraction list, roughly in value order (each its own small PR, post-Review-
Desk-v1 unless noted):

1. **Agent state model + status strip.** Every session exposes a state —
   `working / blocked (awaiting approval or input) / done / failed / idle` —
   derived exactly from the loop (we own it; no output-sniffing like herdr
   needs). Surfaces: color-coded states in the `/agents` switcher, plus a
   one-line strip in the main view when any background session is
   non-idle (`◐ 2 agents: refactor-tests ✔ done · fix-lints ⏸ blocked`).
   herdr's sidebar, shrunk to a single-session product.
2. **Attention signals.** When a session flips to blocked or done: terminal
   bell + OSC title update (`krill ✔ done`), optional macOS notification.
   The user can shove krill code into a background tab/tmux pane and trust
   it to call them — herdr's core promise without herdr.
3. **Multiplexer good-citizenship.** Set pane/window titles (tmux, iTerm,
   herdr) with live state so users who already run a multiplexer get the
   sidebar experience for free. Cheap: same OSC plumbing as #2.
4. **Control socket.** herdr's agent API, re-homed: a local unix socket on
   `krill code` (and/or endpoints on `krill serve`) to list sessions +
   states, inject a prompt, and wait on a session finishing. Makes krill
   scriptable by other tools (Reef, DeepKrill, CI) and is the foundation
   for #5.
5. **Await-able dispatch.** `dispatch_agent` is fire-and-forget today (the
   child never reports back). Add an opt-in `await`/collect mode so an
   agent can spawn a sibling and consume its result — herdr's
   "agents wait on each other", as a first-class harness primitive.
6. **`krill ps` (inference side).** Serve-level visibility: active
   generations, queued requests, which session owns which slot. The
   "who's doing what" question answered at the engine, not the terminal.

Explicitly NOT extracted: pane splitting/layout, PTY hosting of arbitrary
commands, detection heuristics for third-party agents (Claude Code, Codex,
…) — those exist only because herdr doesn't own the agents. We do.
