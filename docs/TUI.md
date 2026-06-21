# Interactive chat TUI

`krill run <model>` with no prompt opens Krill's full-screen chat in the
Sourav AI Labs monochrome identity: a branded masthead, a scrollable
conversation pane, a bottom input box with a slash-command autosuggest popup,
and a status footer (`model . tok/s . ctx N / total (%) . cwd:branch . version`).
It is a multi-turn conversation that remembers context.

It is also an **agent**: type `/agent` to turn hands on (file edits, shell, web
fetch) in the same surface, with `Shift+Tab` cycling how much it may do without
asking. See [Agent mode](#agent-mode) and [Background agents](#background-agents).
`krill code [task]` is an alias that opens this same surface already in agent
mode.

The TUI is the default on an interactive terminal. When stdout is not a TTY
(piped/redirected) Krill falls back to the classic libedit line REPL; force it
with `--classic`. Color is disabled under `NO_COLOR`.

## Themes (light / dark terminals)

Shades adapt to the terminal background so text stays readable on light AND dark
terminals. Resolution order:

1. `--theme light|dark|auto` (default `auto`), or the `KRILL_TUI_THEME` env var.
2. The `COLORFGBG` env var, if the terminal sets it.
3. A best-effort OSC 11 background-color query.
4. An always-safe fallback (relative bold/dim attributes) when none of the above
   resolves.

On a dark terminal the user turn is bright white and the model dim gray; on a
light terminal the user turn is bold (dark-on-light) and the model a medium gray.
If auto-detection ever misreads, pass `--theme light` or `--theme dark`.

## Keys

| Key | Action |
|-----|--------|
| `Up` / `Down` | Recall input history, or cycle the slash-command popup |
| `Tab` | Accept the highlighted command (then add arguments) |
| `Shift+Tab` | Agent mode: cycle the permission posture (plan / ask / accept-edits / auto) |
| `Enter` | Send the message |
| Hold `Space` | Push-to-talk (only when voice is on: dictate / handsfree / send) |
| `Ctrl-V` | Turn voice on and cycle it: off -> dictate -> handsfree -> send -> off |
| `PgUp` / `PgDn` | Scroll the conversation |
| Mouse wheel / trackpad | Scroll the conversation (hold Option/Fn to select text) |
| `Esc` | Interrupt the agent while it is working (in agent mode) |
| `Ctrl-C` | Cancel a streaming reply, or clear the input |
| `Ctrl-D` | Quit |

## Slash commands

Type `/` and an autosuggest popup appears; cycle with Up/Down, run with Enter, or
Tab to fill it and add arguments. `/help` lists everything.

| Command | Action |
|---------|--------|
| `/help` | Show keys and commands |
| `/agent` | Toggle agent mode (tools + file edits); `Shift+Tab` cycles the posture |
| `/bg <task>` | Spawn a background agent for a task (see Background agents) |
| `/agents` | List and switch between background agents (`/switch <n>`, `/main` too) |
| `/config [key=value]` | Show config, or set a key (persists to `~/.krill/config.toml`) |
| `/init` | Generate a `CLAUDE.md` for this repo (runs the agent) |
| `/diff` | Show pending working-tree changes (`git diff`) |
| `/status` | Show version, model, working directory, posture, context |
| `/context` | Show context-window usage for the last turn |
| `/copy` | Copy the last reply to the clipboard |
| `/cd <path>` / `/add-dir <path>` | Change / add a working directory the agent can use |
| `/model [name]` | Open the model picker, or switch/download a named model in place |
| `/model info [name]` | Open the model deep-dive (also `i` / right-arrow on a row in the picker) |
| `/system <text>` | Set the system prompt |
| `/history` | Print the turns so far |
| `/compact` | Summarize the conversation and replace history with the summary (frees context) |
| `/save [file]` | Write the transcript to a file |
| `/clear` | Clear the conversation (`/reset` is a hidden alias) |
| `/image <path>` / `/audio <path>` | Attach media to your next message (`/img` too) |
| `/attach` | List pending attachments (index, dimensions, size) |
| `/remove <n>` | Drop attachment number n |
| `/drop` | Drop all pending attachments |
| `/mic` | Record from the microphone (press Enter to stop) |
| `/voice-mode type\|dictate\|handsfree\|send` | Set the voice posture (Ctrl-V cycles; see Voice) |
| `/voice` | Show the current voice state (posture + engine) |
| `/voice engine apple\|whisper` | Choose the dictation engine (see Voice) |
| `/quit` | Exit (`/exit`, `/q` too) |

## Agent mode

`/agent` turns the chat into an **agent**: the model gains tools and can act on
your project. Type `/agent` again to turn hands back off. `krill code [task]`
opens the same surface already in agent mode (and runs `task` if given).

The toolset: read-only explorers (`read_file`, `list_dir`, `glob`, `grep`),
`web_search` (search the web for links + snippets) and `web_fetch` (fetch a URL
as readable text), file edits (`edit_file`, `multi_edit`, `write_file`), `bash`,
and `dispatch_agent` (spawn a background agent). As it works, the transcript
shows each step as an action chip
(`▸ edit_file path`), the tool's result (with a `+N -M` diffstat on edits), and a
live footer (`working . 8s . Esc interrupt`). Press **Esc** (or `Ctrl-C`) to
interrupt a run.

### Permission postures

What the agent may do without asking is the **posture**, cycled live with
**`Shift+Tab`** and shown as a footer chip (`agent:plan`). Read-only tools always
run; the posture governs the mutating ones:

| Posture | Behaviour |
|---------|-----------|
| `plan` | Read-only. The agent investigates and proposes a plan; edits and commands are denied. |
| `ask` | Confirm every file edit and shell command before it runs. |
| `accept-edits` | File edits apply automatically; shell commands still ask. |
| `auto` | Everything runs without asking. |

In `ask` / `accept-edits`, when the agent wants to run a gated tool an approval
bar appears above the input box: `[y]es` runs it, `[n]o` (or `Esc`) denies it,
`[a]lways` allows that tool for the rest of the session.

The launch defaults come from `~/.krill/config.toml`: `default_mode`
(`chat` or `agent`) and `default_agent_posture` (`plan` / `ask` / `accept-edits`
/ `auto`). Set them in place with `/config default_mode=agent`.

## Background agents

Spawn agents that run independently and switch between them - Krill's analogue of
Claude Code's background sessions (not invisible subagents).

- **`/bg <task>`** starts a background agent on `task` (it inherits the current
  model and posture). The model can also spawn one itself with the
  `dispatch_agent` tool, e.g. to fan a sub-task out while it keeps working.
- **`/agents`** opens a switcher listing the main view plus every background
  agent with its live status; `Enter` attaches. `/switch <n>` and `/main` (return
  to the main view) do the same without the popup.
- While **attached** you watch the agent live, answer its approval prompts
  (`y`/`n`/`a`), `Esc` to interrupt it, or - once it is idle - type to continue
  its conversation. The footer shows a background-agent count (with a `!` when one
  is waiting on an approval).

Because the in-process path has a single shared model, generations are
serialized: agents progress turn by turn and never decode at the same instant
(there is no throughput gain from running several at once on one GPU; the value
is being able to watch, steer, and switch between them).

## Web search

`web_fetch` reads a page you already have a URL for; `web_search` finds the URLs.
The agent searches, gets a ranked list of titles/URLs/snippets, then fetches the
promising ones to read them.

Search is **off until you point Krill at a backend** - it is local-first, with no
API key. Today the backend is a self-hosted [SearXNG](https://docs.searxng.org/)
instance:

```
/config searxng_url=http://localhost:8888
```

(or export `KRILL_SEARXNG_URL`). The instance must have `json` enabled in its
`search.formats` - SearXNG ships HTML-only, so add `json` to that list in its
`settings.yml`. The backend is pluggable behind `search_backend` (default
`searxng`); when no backend is configured the tool returns a one-line note
telling you how to enable it rather than failing silently.

### Deep research

`/research <question>` runs a multi-step research pass and writes a cited answer
into the conversation:

1. **Plan** - the model turns your question into a few focused search queries.
2. **Search** - each query goes to the search backend; the results are pooled and
   de-duplicated by URL.
3. **Read** - the top sources are fetched with `web_fetch` and each page is
   summarized on its own (only the short summary is kept, so a long page never
   blows the context window).
4. **Synthesize** - the model writes the answer from those summaries, citing
   sources inline as `[1]`, `[2]`, ... with a `Sources:` list at the end.

It runs in the foreground with a live progress trail (planning -> searching ->
reading [n/total] -> synthesizing); press `Esc` or `Ctrl-C` to stop. The search
and fetch steps are driven by code, not by the model agentically - each model
call is a single bounded step, which is far more reliable on a small local model
than asking it to drive a long tool loop itself. Needs `searxng_url` set (same as
`web_search`); without it `/research` tells you how to enable search. The answer
is added to the conversation, so you can ask follow-ups that build on it.

## Model deep-dive

In the model picker, press `i` (or the right-arrow) on a row - or run
`/model info [name]` - to open a deep-dive screen for that model: a stylized
family wordmark, the live specs (parameters, quantization, context window,
on-disk/download size, supported inputs, features, repo), and a short curated
profile (vendor, release, strengths, weaknesses, and what it is good for). The
specs are derived from the registry and capability metadata; the profile is a
hand-written blurb per family. Esc closes it.

## Custom slash commands

Drop a Markdown file at `~/.krill/commands/<name>.md` and it becomes `/<name>`,
the Krill analogue of Claude Code's `.claude/commands`. The body is a prompt
template expanded with the text typed after the command, then sent like a normal
message. Custom commands appear in the autosuggest popup and in `/help`.

Placeholders:

- `$ARGUMENTS`, `$ARGS`, `$INPUT` -> the whole argument string
- `$1` .. `$9` -> whitespace-split positional words
- A template with no placeholder gets the arguments appended on a new line

An optional `--- description: ... ---` frontmatter block sets the one-line
summary shown in the popup. Example `~/.krill/commands/tldr.md`:

```markdown
---
description: Summarize in three bullets
---
Summarize the following in exactly three concise bullet points:

$ARGUMENTS
```

Then `/tldr <text>` expands and runs. Built-in commands always win on a name
clash, so a custom command cannot shadow `/model`, `/clear`, etc.

## Attachments

Attach images and audio without leaving the session, three ways:

```text
> /image ~/Pictures/cat.png            # explicit command (/audio, /img too)
> /Users/me/My Photos/cat.png          # drag a file into the terminal (path is pasted)
> what breed is @~/Pictures/cat.png?   # inline @path inside your message
```

Attachments apply to your **next** message, then clear. Images accumulate for
multi-image models (mllama); single-image models use the first. `--image` /
`--audio` on the command line pre-attach to the first turn.

## Voice

**Voice is off by default** - Krill is a text chat first. In the default
**`type`** posture Space is a typed space, Enter sends, and the footer shows no
voice chrome. Turn voice on with **`Ctrl-V`** (which then cycles the postures) or
set a default with the `voice_mode` config key (`off` / `dictate` / `handsfree`).
On an audio-capable Gemma 4 model the active posture rides the footer's left side
(a dot when on, an animated meter while recording). Bare `/voice` prints the
current state without changing it.

`Ctrl-V` cycles **off (text) -> dictate -> handsfree -> send -> off**.

- **`type`** (default) - keyboard only. **Space is a typed space** and Enter
  sends; there is no push-to-talk. The footer stays clean.
- **`dictate`** - hold **Space** to talk; your speech is transcribed into the
  composer for you to **review and send**. Uses the chosen engine (Apple or
  Whisper; see below).
- **`handsfree`** - hold Space to talk; the transcript is shown and then
  **auto-sent** after a short grace window (press **Esc** to cancel, **Enter** to
  send immediately). The reply is shown on screen; spoken replies (TTS) are a
  planned follow-up.
- **`send`** - the clip is sent as an audio turn and the model **answers your
  spoken input** (shown as a `[voice message]` turn). Use this to "talk to" an
  audio-capable model rather than dictate.

### Dictation engine

`/voice engine` (no argument) prints a card showing the current choice;
`/voice engine apple|whisper` switches it:

- **`apple`** (default) - Apple's on-device speech-to-text. No download,
  instant, fully local, macOS-only.
- **`whisper`** - Krill's own native MLX Whisper runtime. Higher accuracy and
  fully local. On the first dictation it asks consent and downloads a model
  (default `base.en`, around 290 MB) into `~/.krill/models/whisper-<sku>`;
  decline and dictation falls back to the Apple / model path. No Python or
  third-party ASR dependency.

  Pick the model with `/voice engine whisper <model>`:
  - `tiny.en` / `base.en` / `small.en` - English-only (faster, slightly more
    accurate on English).
  - `tiny` / `base` / `small` - multilingual (~99 languages with automatic
    language detection).

`/mic` records a clip and attaches it (press Enter to stop) for an explicit send.

macOS attributes microphone access to the running app, so the mic needs Krill
to run from a code-signed bundle that declares the mic-usage string:

```bash
make app-bundle                 # produces dist/krill.app (ad-hoc signed)
dist/krill.app/Contents/MacOS/krill run gemma-4-e2b
```

The first mic use triggers the system permission prompt under Krill's identity.
Run the bare `.build/release/krill` binary instead and the prompt attaches to
your terminal app.
