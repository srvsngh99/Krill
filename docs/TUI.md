# Interactive chat TUI

`krillm run <model>` with no prompt opens KrillLM's full-screen chat in the
Sourav AI Labs monochrome identity: a branded masthead, a scrollable
conversation pane, a bottom input box with a slash-command autosuggest popup,
and a status footer (`model . tok/s . ctx N / total (%) . cwd:branch . version`).
It is a multi-turn conversation that remembers context.

The TUI is the default on an interactive terminal. When stdout is not a TTY
(piped/redirected) KrillLM falls back to the classic libedit line REPL; force it
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
| `Enter` | Send the message |
| `PgUp` / `PgDn` | Scroll the conversation |
| Mouse wheel / trackpad | Scroll the conversation (hold Option/Fn to select text) |
| `Ctrl-C` | Cancel a streaming reply, or clear the input |
| `Ctrl-D` | Quit |

## Slash commands

Type `/` and an autosuggest popup appears; cycle with Up/Down, run with Enter, or
Tab to fill it and add arguments. `/help` lists everything.

| Command | Action |
|---------|--------|
| `/help` | Show keys and commands |
| `/model [name]` | Open the model picker, or switch/download a named model in place |
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
| `/voice send\|dictate` | Switch what hold-Space does (see Voice) |
| `/voice engine apple\|whisper` | Choose the dictation engine (see Voice) |
| `/quit` | Exit (`/exit`, `/q` too) |

## Custom slash commands

Drop a Markdown file at `~/.krillm/commands/<name>.md` and it becomes `/<name>`,
the KrillLM analogue of Claude Code's `.claude/commands`. The body is a prompt
template expanded with the text typed after the command, then sent like a normal
message. Custom commands appear in the autosuggest popup and in `/help`.

Placeholders:

- `$ARGUMENTS`, `$ARGS`, `$INPUT` -> the whole argument string
- `$1` .. `$9` -> whitespace-split positional words
- A template with no placeholder gets the arguments appended on a new line

An optional `--- description: ... ---` frontmatter block sets the one-line
summary shown in the popup. Example `~/.krillm/commands/tldr.md`:

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

On an audio-capable Gemma 4 model, hold **Space** on an empty composer to talk
(push-to-talk); release to finish. What happens with the clip depends on the
mode, toggled with `/voice`:

- **`/voice dictate`** (default) - transcribes your speech into the composer for
  you to review and send. Uses **Apple's on-device speech-to-text** (fully local,
  no download) when available; otherwise falls back to the multimodal model's
  best effort (which tends to *answer* rather than transcribe).
- **`/voice send`** - the clip is sent as an audio turn and the model answers
  your spoken input (shown as a `[voice message]` turn). Use this to "talk to"
  an audio-capable model rather than dictate.

### Dictation engine

`/voice engine` (no argument) prints a card showing the current choice;
`/voice engine apple|whisper` switches it:

- **`apple`** (default) - Apple's on-device speech-to-text. No download,
  instant, fully local, macOS-only.
- **`whisper`** - KrillLM's own native MLX Whisper runtime. Higher accuracy and
  fully local. On the first dictation it asks consent and downloads a model
  (default `base.en`, around 290 MB) into `~/.krillm/models/whisper-<sku>`;
  decline and dictation falls back to the Apple / model path. No Python or
  third-party ASR dependency.

  Pick the model with `/voice engine whisper <model>`:
  - `tiny.en` / `base.en` / `small.en` - English-only (faster, slightly more
    accurate on English).
  - `tiny` / `base` / `small` - multilingual (~99 languages with automatic
    language detection).

`/mic` records a clip and attaches it (press Enter to stop) for an explicit send.

macOS attributes microphone access to the running app, so the mic needs KrillLM
to run from a code-signed bundle that declares the mic-usage string:

```bash
make app-bundle                 # produces dist/krillm.app (ad-hoc signed)
dist/krillm.app/Contents/MacOS/krillm run gemma-4-e2b
```

The first mic use triggers the system permission prompt under KrillLM's identity.
Run the bare `.build/release/krillm` binary instead and the prompt attaches to
your terminal app.
