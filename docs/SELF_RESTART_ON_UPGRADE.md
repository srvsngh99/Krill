# Self-restart on upgrade

A running `krill serve` keeps executing the old image after the file on disk is replaced. When the
server runs as a launchd job it now notices the new version and restarts itself, so
`brew upgrade krill` is the only command you run. Covers `krill serve` jobs and the `krill ui --install`
login item. The TUI and `krill code` are interactive and are not affected.

## How it works

1. **Watch.** About every 60 s the server `stat`s the path it was launched as (`argv[0]` made absolute,
   not the resolved target) and compares *resolved path + inode + mtime + size* with what it recorded at
   start. Homebrew repoints `/opt/homebrew/bin/krill` to a new Cellar directory (the resolved path
   changes); `install.sh` does `rm` + `cp` behind `/usr/local/bin/krill` (inode and mtime change).
2. **Vet the candidate.** A change counts only if it is unchanged on a second check one interval later
   (a half-written file keeps changing), is an executable file, passes `codesign --verify` (macOS) and
   `krill version` answers with a version different from the running one (90 s limit each; a timeout on a busy
   machine is retried at the next check, not held against the file). Anything else is ignored and logged once.
3. **Wait until idle.** No queued or running generation, embedding or rerank request
   (`GenerationQueue.pending`) and no request head seen for 10 s. While waiting it logs once, e.g.
   `restart to 0.27.0 deferred: waiting for 2 in-flight request(s)`.
4. **Restart.** Stop accepting, drain, unload (the same path as SIGTERM, `ShutdownSignals`), then run
   `launchctl kickstart -k gui/<uid>/<label>`. launchd itself stops and relaunches the job onto the
   binary now on disk. An exit code alone would not do: exit 0 does not relaunch under
   `KeepAlive = { SuccessfulExit = false }` and nothing relaunches without `KeepAlive`. If `kickstart`
   fails the process exits 75, which relaunches under `KeepAlive = true` and `SuccessfulExit = false`.

"Under launchd" means `XPC_SERVICE_NAME` is a job label (not `0` / `application.*`, which terminals
carry) and the parent pid is 1. A terminal `krill serve` never restarts: it prints and logs one notice,
`Krill X.Y.Z is installed; restart to use it`.

## Knobs

| Knob | Default | Effect |
|---|---|---|
| `auto_restart_on_upgrade` (config.toml) | `true` | `false` makes launchd servers notice-only too. |
| `KRILL_AUTO_RESTART_ON_UPGRADE` | unset | `0`/`false`/`no`/`off` disable, anything else enables; beats the config key. |
| `KRILL_UPGRADE_CHECK_SECONDS` | `60` | Poll interval (tests). |
| `KRILL_UPGRADE_QUIET_SECONDS` | `10` | Required request-free period (tests). |

To turn it off for one job, add `KRILL_AUTO_RESTART_ON_UPGRADE=0` to the plist's
`EnvironmentVariables`; globally, put `auto_restart_on_upgrade = false` in `~/.krill/config.toml`.

## Notes

- `krill ui --install` now records the stable `bin/krill` path in its login item when krill lives in a Homebrew
  Cellar (the Cellar dir is deleted by `brew upgrade`). Re-run `krill ui --install` once to refresh an old plist.
- Log lines are at notice level with the old and new version (`~/.krill/ui/serve.log` for the UI login
  item, your plist's `StandardOutPath` otherwise).
- Only a `gui/<uid>` LaunchAgent can be kickstarted; a LaunchDaemon falls back to exit 75.
- Not covered: agent-session turns that hold no generation slot do not defer a restart beyond the
  10 s quiet period.
