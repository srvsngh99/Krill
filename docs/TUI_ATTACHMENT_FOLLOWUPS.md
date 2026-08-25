# TUI attachment follow-ups (from PR #305)

Work order for the two items consciously left out of PR #305 ("fix: attach TUI
images and add native copy mode"). #305 fixed the reported bug — a dragged image
path was sent to the model as prose instead of being attached — and added copy
mode. These two items are the *next* failures a user will hit on the same
surface. Neither is a regression; both are pre-existing shapes that #305
inherited or propagated.

Read this together with `docs/BACKLOG.md` (short entries) — this file is the
implementable version.

> **Status: implemented by PR #307.** Both items below shipped, sharing one
> scanner (`MediaAttachment.extractReferences`) and one two-stage loader
> (`MediaAttachment.readMediaFile`) across `ChatTUI` and the line REPL. This
> file is kept as the spec of record — the reasoning, the rejected
> alternatives, and the testability constraint that shaped the implementation.
> One known limit remains: `detectKind` falls back to the file extension, so a
> large file *named* `.png` still passes the 64-byte gate and is read in full.
> The size ceiling that closes it is tracked in `docs/BACKLOG.md`.

**Line references** are against `codex/fix-tui-attachments-copy-mode` @ `8244535`
(i.e. post-#305). They will shift once other PRs land; grep the symbol names.

---

## Item 1 — attach a bare media path that appears MID-SENTENCE

### Problem

After #305, a bare path attaches only when it is the **entire** message:

```swift
// Sources/KrillCLI/TUI/ChatTUI.swift:905
if atts.isEmpty {
    switch loadMedia(trimmed) {   // `trimmed` == the whole submitted line
```

So this works:

```
/Users/me/shot.png                       → attached
```

and this does not:

```
what's wrong in this screenshot? /Users/me/shot.png    → sent as prose
```

The second shape produces exactly the failure #305 set out to fix: the model
receives a path it cannot read, guesses, and burns turns on `read_file` /
`bash` against a file that was never attached. In plan mode the `bash` attempt
is correctly denied, which makes the transcript look like a permissions problem
rather than an attachment problem.

This is parity with the line REPL (`InteractiveSession.swift:179`), which has
the same whole-line-only restriction — so #305 is not a regression. It is,
however, the obvious next report.

### Spec

Extend `extractInline` to recognize a **bare** path token anywhere a `@path`
token would be recognized, subject to a cheap guard so ordinary prose is never
probed against the filesystem.

Rules:

1. A token is a candidate only at a word boundary (`atBoundary`, already
   tracked) **and** only if it starts with `/`, `~`, or `.` — or is `@`-marked
   (existing behavior). Prose words never hit the filesystem.
2. Honour `\`-escapes while scanning the token, so a Terminal drag
   (`Screenshot\ 2026-08-24\ at\ 9.58.29\ AM.png`) stays one token. The existing
   `@`-scanner already does this; factor it into a shared local function rather
   than duplicating it.
3. A candidate that resolves via `loadMedia` is removed from the prompt text and
   attached. A candidate that does not resolve is left in the text verbatim —
   never swallowed.
4. `.unsupported` (real media, wrong model) must produce a **visible note**, not
   silent pass-through. Silent pass-through is what makes the model improvise.

### Ordered steps

1. In `Sources/KrillCLI/TUI/ChatTUI.swift`, change `extractInline` (`:3136`) to
   return a third value: `(String, [Att], [String])` — cleaned text,
   attachments, and user-facing notes.
2. Inside it, extract the token scanner into a local
   `func token(from start: Int) -> (text: String, end: Int)` honouring `\`
   escapes, and use it for both the `@` and bare branches.
3. Gate the bare branch on `atBoundary && (ch == "/" || ch == "~" || ch == ".")`.
4. Switch on `loadMedia(tok)`: `.ok` attaches and consumes the token;
   `.unsupported(kind)` appends a note naming the file and the reason and leaves
   the text in place; `.notFound` / `.notMedia` fall through untouched.
5. Update the caller in `processSubmit` (`:890`) to destructure three values and
   append each note as a `Msg(role: .note, …)`.
6. Delete the now-redundant whole-line `if atts.isEmpty { switch loadMedia(trimmed) … }`
   block at `:905` — the general scanner subsumes it. Verify the whole-line
   drag case still attaches (it is just the single-token case).
7. Mirror the same change in `InteractiveSession.extractInlineMedia`
   (`:377`) so the two surfaces do not drift again. The whole-line branch at
   `:179` becomes redundant there too.

### Verification

- Unit: the scanner is pure once extracted. Move it to a testable location (see
  "Testability" below) and cover: bare path mid-sentence; `@path` mid-sentence;
  escaped spaces; a path that does not exist (left in text); prose containing
  `.` and `/` (e.g. "3.5", "and/or") that must NOT hit the filesystem; an
  unsupported-kind path producing a note.
- Manual, both surfaces (`krill run <vision-model>` and `krill code`):
  1. Drag an image, press Enter with no other text → attaches (regression guard).
  2. Type `what is in this? ` then drag the image → attaches AND the prompt text
     survives without the path.
  3. Same on a text-only model → visible note, no silent drop.

### Rejected alternatives

- **Probe every whitespace token against the filesystem.** Simplest, but it
  `stat`s on every word of every message and would attach a file for a message
  like `README.md`. The `/`/`~`/`.` prefix guard keeps the cost at zero for
  prose.
- **Only support `@path` and document it.** This is what the code claimed to do
  before #305, and the drag gesture — which inserts a bare path with no `@` —
  is the single most common way users attach an image. Documenting around the
  gesture does not fix the gesture.
- **Strip the path from the prompt and pass it as a tool arg.** Larger change,
  and it hands the model a path instead of pixels, which is the failure mode
  being removed.

---

## Item 2 — sniff the header before reading the whole file

### Problem

```swift
// Sources/KrillCLI/TUI/ChatTUI.swift:3092
private func loadMedia(_ token: String) -> Load {
    …
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue,
          let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return .notFound }
    let ext = (path as NSString).pathExtension
    guard let kind = MediaAttachment.detectKind(data: data, pathExtension: ext) else { return .notMedia }
```

The entire file is read into memory *before* anything checks whether it is
media. Submit a bare path to a multi-GB file and Krill slurps it all to then
return `.notMedia`.

`MediaAttachment.detectKind` never inspects past the first **16 bytes**
(`Sources/KrillCore/MediaAttachment.swift` — `isImageData` reads at most 12,
`isAudioData` takes `data.prefix(16)`), so the full read buys nothing at the
detection step.

Severity is low today because it needs a bare path to a huge file as the whole
message. It rises sharply with **Item 1**, which makes every `/`-prefixed token
in any sentence a candidate. Ship Item 2 **before or with** Item 1, not after.

Context: this box has a ~14GB working ceiling and the repo lives on an
iCloud-synced Desktop, so an accidental multi-GB read is not academic.

### Spec

Two-stage read: sniff a 64-byte header to decide *whether* this is media, and
only then read the file for real.

### Ordered steps

1. In `loadMedia` (`ChatTUI.swift:3092`), after the `fileExists` guard, replace
   the single `Data(contentsOf:)` with:
   - open a `FileHandle(forReadingAtPath:)`, `read(upToCount: 64)`, close it;
   - `guard MediaAttachment.detectKind(data: head, pathExtension: ext) != nil
     else { return .notMedia }`;
   - only then `Data(contentsOf:)` and re-run `detectKind` on the full data to
     get the definitive `kind`.
2. Keep the second `detectKind` — do not trust the 64-byte result as final. The
   header sniff is a cheap *reject* filter; the full-data call stays
   authoritative so behavior is identical for everything that is media.
3. Apply the identical change to `InteractiveSession.loadMedia` (`:358`).
4. Consider a size ceiling as a separate guard (e.g. refuse to attach >64MB with
   a clear note). Out of scope here; note it in `docs/BACKLOG.md` if not done.

### Verification

- Unit: `MediaAttachment.detectKind` already has coverage; add a case asserting
  a 64-byte PNG header alone is detected as `.image`, which is the invariant the
  optimization leans on.
- Manual: `mkfile 3g /tmp/big.bin`, submit `/tmp/big.bin` as an entire message,
  confirm it returns "not a recognized image or audio file" **immediately** and
  RSS does not spike. Then confirm a real PNG still attaches.

### Rejected alternatives

- **Extension-only check before reading.** `detectKind` deliberately sniffs
  magic bytes first so an extensionless dragged file still works; gating on the
  extension would regress that.
- **Memory-map the file.** More machinery than a 64-byte read for no additional
  benefit at the detection step.

---

## Testability (applies to both items)

`ChatTUI` lives in the `krill` **executable** target, so none of this is
reachable from the test bundle — see the `KrillLaunch` precedent and the
`--test-bundle-path` CI failure that motivated it.

To gate either item, move the pure logic into a library target:

- `extractInline`'s scanner → a free function over `(String, (String) -> Load)`
  in `KrillCore` (or `KrillTUI`), injecting the resolver so tests never touch
  the filesystem.
- the header-sniff decision → already library code in `KrillCore`
  (`MediaAttachment`); only the call site moves.

Without that extraction these can only be verified manually, which is how the
original bug survived: the attachment path had no test that used a *bare* path.
