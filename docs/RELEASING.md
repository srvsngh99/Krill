# Releasing Krill

Two things live here: **what is merged but not yet shipped** (read this first when
cutting a release), and **how to cut one**.

The canonical, categorized history is [`CHANGELOG.md`](../CHANGELOG.md); its
`[Unreleased]` section and the ledger below must agree. The one-blurb-per-version
view is [`RELEASES.md`](../RELEASES.md).

---

## Merged, awaiting release

Work that is on `main` and will ship in the next version. **Clear this table as
part of cutting a release** — move the entries into the new version's CHANGELOG
section, then delete the rows.

| Merged | PR | What | Why it matters at release time |
|---|---|---|---|

_Nothing pending._

### Known-open, NOT yet merged

Tracked so a release note does not claim more than shipped.

- **Plan mode denies `bash` wholesale.** `Permission.decision(toolName:isReadOnly:isFileEdit:)`
  never sees the tool arguments, so a read-only command (`find . | wc -l`) is
  denied by the same rule as `rm -rf`. Fixing it means a per-call read-only
  classifier for shell commands — a security boundary, since plan mode's whole
  promise is "read-only". Needs an adversarial test suite (`find -delete`,
  `find -exec`, `git diff > f`, backticks, `$( )`, `;`, `&&`, `sed -i`) before it
  can be trusted. See the trap table in the PR discussion.
- **`reasoning_effort` has no `krill config` knob.** `KRILL_REASONING_EFFORT` is
  the only surface. Wiring the config key without the TUI toggle would create a
  dead setting, so both land together or neither does.

---

## Cutting a release

Proven flow (v0.18.0 → v0.21.0). **One PR**, not a separate SHA-fix PR.

```bash
# 0. Start clean, from an up-to-date main
git checkout main && git pull && git status   # must be clean
git checkout -b release/vX.Y.Z
```

1. **Clear the ledger above** — move "Merged, awaiting release" rows into the new
   CHANGELOG section, then empty the table.
2. **Bump the version in both places** — `VERSION` and
   `Sources/KrillRegistry/KrillVersion.swift`. `KrillVersionTests` asserts they
   match, so a mismatch fails the build, not production.
3. **Write the notes** — `CHANGELOG.md` (categorized: Added / Fixed / Changed)
   and `RELEASES.md` (one human blurb: what changed for a user, not a commit
   list).
4. **Rebuild — do not skip this.**
   ```bash
   swift build -c release --arch arm64
   make metallib CONFIGURATION=release
   ./.build/arm64-apple-macosx/release/krill version   # MUST print the new number
   ```
5. **Gate the public build.** Kreach is private and must never ship:
   ```bash
   strings -a .build/arm64-apple-macosx/release/krill | grep -ic kreach   # must be 0
   ```
6. **Package and stamp.**
   ```bash
   make dist CONFIGURATION=release
   # put the printed sha256 into Formula/krill.rb (url + version too)
   ```
   Verify the tarball is what you think it is — extract it and run
   `krill version` from the extracted binary.
7. **One PR**, wait for CI green, merge.
8. **Tag and publish**, using that exact tarball:
   ```bash
   git checkout main && git pull
   git tag vX.Y.Z && git push origin vX.Y.Z
   gh release create vX.Y.Z dist/krill-X.Y.Z-arm64-apple-macos.tar.gz --title ... --notes ...
   ```
9. **Cross-check three digests**: local tarball == formula `sha256` ==
   published asset `digest` (`gh release view vX.Y.Z --json assets`).
10. **Update the tap** — copy `Formula/krill.rb` to
    `/opt/homebrew/Library/Taps/srvsngh99/homebrew-krill`, branch, PR, merge.
11. **Live-verify the published path**, not the local build:
    ```bash
    KRILL_PREFIX=/tmp/krillcheck KRILL_VERSION=X.Y.Z \
      sh <(curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh)
    /tmp/krillcheck/bin/krill version
    ```

---

## Traps that have actually bitten

- **`make dist` does not rebuild.** It only tars `.build/release`. Bump
  `KrillVersion.swift` and rebuild *first*, or you ship a binary that reports the
  previous version. Extracting the tarball and running `krill version` catches it.
- **Any source change after `make dist` invalidates the SHA.** Re-run `make dist`
  and re-stamp the formula **in the same PR**. v0.19.0 needed this when CI
  rejected a trailing blank line at EOF.
- **`make metallib` defaults to DEBUG.** Pass `CONFIGURATION=release` or the
  tarball ships the wrong metallib.
- **Version numbers only go up.** Brew and `krill update` semver-compare against
  the published version, so a "0.16.4" published after "0.17.0" can never be
  picked up. There is no un-publishing.
- **The tap's `main` is hook-protected** and enforces the author identity
  `Sourav Singh <srv.sngh99@gmail.com>`. Branch and PR there too; never
  `--no-verify`.
- **Model-blob aliases must point at a repo that exists.** An `AliasMap` entry
  referencing an unpublished HF repo builds and tests green, then 404s for every
  user. Publish the weights before the release, and confirm the file count and
  total size match the source.
