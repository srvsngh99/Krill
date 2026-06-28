# 0003. Persistent Memory (opt-in, text + semantic, BYOM)

Status: proposed. Date: 2026-06-28. Owner: Sourav. Scope: how Krill gives users
long-term memory across sessions — what is stored, where, in what form, how the
model reaches it, and why this is built natively in Swift rather than bolted on
with an external vector database.

---

## 0. TL;DR

- Memory is **opt-in and off by default** (`memory_backend = off`). A fresh
  install behaves exactly as today; nothing is written to disk until the user
  turns it on.
- Two **flavours** behind one `MemoryStore` protocol (the `SearchBackend`
  pattern, see [0002](0002-web-search-backends.md)):
  - `text` — plain append-only JSONL, keyword/recency recall, **zero new deps**
    (no MLX, no Python).
  - `semantic` — same JSONL **plus** an embedding cache for cosine recall,
    reusing the existing `EmbeddingEngine` (`KrillEngine/EmbeddingEngine.swift`).
- **No external vector DB (no ChromaDB).** Krill is a self-contained Swift/MLX
  binary with **no Python runtime**; semantic search uses the embedder we already
  ship + brute-force cosine. See §3 for the full rejection.
- **The store survives the release lifecycle.** Text is the source of truth;
  vectors are a *rebuildable cache*. No Krill command (install, upgrade,
  uninstall, prune) ever deletes `memory/`. A release can, at worst, trigger a
  one-time re-embed — never data loss.
- **Bring-your-own-memory.** Default dir is `~/.krill/memory/`; `memory_dir` /
  `KRILL_MEMORY_DIR` repoint it at any folder the user owns (iCloud, Dropbox, a
  git repo, a NAS).
- The model reaches memory two ways: **tools** (`memory_save` / `memory_search` /
  `memory_list`, Phase 1) and optional **auto-recall** context injection at
  session start (`memory_auto_recall`, Phase 2, off by default).

## 1. Outcomes we want

1. A Krill user can ask it to remember things and have them recalled in later,
   unrelated sessions — the thing that makes a local agent feel like *theirs*.
2. Turning memory on is a single explicit choice. Users who never opt in pay
   **zero** cost: no disk writes, no extra model in RAM, no new tools, no
   behavioural change.
3. Two levels of capability so users pick their trade-off: a dependency-free
   plain-text memory anyone can read/grep/edit/commit, and a semantic memory
   with meaning-based recall.
4. **Durability is a contract.** A user's memory must outlive any number of
   `krill update`s, a reinstall, even an uninstall. No release may corrupt or
   delete it. The format must be inspectable and forward-compatible.
5. It stays true to Krill's shape: one self-contained Swift binary, Apple-Silicon
   native, installable with one command, no new runtime.
6. Memory is *portable and user-owned* — a directory the user can move, sync,
   back up, or hand to another tool.

### Non-goals (explicitly out of scope here)

- Cloud sync / multi-device replication (BYOM + a synced folder covers the common
  case without us operating a service).
- Multi-user or shared/team memory.
- At-rest encryption (the dir inherits the user's FileVault/disk posture; can be a
  future record if demanded).
- Memory *editing UX* beyond list/save/search (the JSONL is hand-editable today).

## 2. Decision

### 2.1 One protocol, three conformers (mirror `SearchBackend`)

A `MemoryStore` protocol in `KrillHarness` (alongside `Tool` and `SearchBackend`):

```swift
public struct MemoryEntry: Sendable, Equatable {
    public let id: String          // content-derived or UUID; stable
    public let text: String        // the memory — source of truth
    public let tags: [String]
    public let createdAt: String   // ISO-8601 string (no Date.now on deterministic paths)
    public let source: String?     // "chat" | "user" | session id
}
public struct MemoryHit: Sendable { public let entry: MemoryEntry; public let score: Double }

public protocol MemoryStore: Sendable {
    var name: String { get }                                  // "off" | "text" | "semantic"
    func save(_ entry: MemoryEntry) async throws
    func search(query: String, limit: Int) async throws -> [MemoryHit]
    func recent(limit: Int) async throws -> [MemoryEntry]
}
```

| Conformer | `memory_backend` | Recall | Deps |
|---|---|---|---|
| `NoMemoryStore` | `off` (default) | none — `search`/`recent` return `[]`, `save` is a no-op | none |
| `TextFileMemoryStore` | `text` | recency + keyword/substring scoring over `entries.jsonl` | none (no MLX) |
| `SemanticMemoryStore` | `semantic` | cosine over cached embeddings; **falls back to keyword** if the embedder is unavailable | `EmbeddingEngine` via an injected `MemoryEmbedder` |

A `MemoryStore.configured(embedder:)` factory selects the conformer from
`KrillConfig`, exactly as `WebSearchTool.configuredBackend()` does today
(`KrillHarness/Tools/WebSearchTool.swift:159`).

### 2.2 Dependency direction (keeps MLX off the text path)

The protocol, `NoMemoryStore`, and `TextFileMemoryStore` live in `KrillHarness`
with **no `KrillEngine` dependency**. `SemanticMemoryStore` depends only on a tiny
local protocol:

```swift
public protocol MemoryEmbedder: Sendable { func embed(_ texts: [String]) throws -> [[Float]] }
```

The concrete `EmbeddingEngine`-backed embedder is constructed and **injected at the
composition root** (`KrillCLI` / `KrillAgent`, which already link `KrillEngine`).
So MLX is linked into the memory path only when semantic memory is actually wired,
and the embedder **loads lazily** — only when `memory_backend = semantic`.

### 2.3 On-disk format — text is truth, vectors are cache

```
<memory_dir>/                         # default ~/.krill/memory/
  entries.jsonl                       # SOURCE OF TRUTH — one JSON object per line, append-only
  index/
    meta.json                         # { embedModel, dim, schemaVersion }
    vectors.bin                       # REBUILDABLE CACHE — regenerated if meta mismatches
```

- `text` flavour writes only `entries.jsonl`.
- `semantic` flavour adds `index/`. On load, if `meta.json` doesn't match the
  current embed model / dimension / schema (or `vectors.bin` is missing or
  short), Krill **silently re-embeds from `entries.jsonl`**. This is the
  mechanism that makes durability a contract: a release can change the embedder
  or bump the schema and the worst outcome is a one-time re-embed.

### 2.4 Storage location & BYOM

Resolved in order: `KRILL_MEMORY_DIR` env → `memory_dir` config → default
`~/.krill/memory/`. The directory is **append-only from Krill's side**; no Krill
code path deletes it (see §5 for the enforced guard). BYOM = point `memory_dir`
at any folder the user owns.

### 2.5 How the model reaches memory

- **Phase 1 — tools** (registered only when `memory_backend != off`):
  - `memory_search(query, limit)` — `isReadOnly = true`, never prompts.
  - `memory_save(text, tags?)` — appends to the user-owned dir.
  - `memory_list(limit)` — recent entries.
- **Phase 2 — auto-recall** (behind `memory_auto_recall`, default off even once
  memory is on): at session start, embed the opening user turn, retrieve top-K
  relevant entries, and prepend them as context — the "it just remembers"
  experience.

### 2.6 Config surface (mirrors the search keys)

```toml
memory_backend     = "off"            # off | text | semantic   (default: off)
memory_dir         = "~/.krill/memory" # BYOM override; KRILL_MEMORY_DIR wins
memory_embed_model = "bge-small-en"    # semantic only; embedder from the registry
memory_max_results = 5
memory_auto_recall = false             # Phase 2; opt-in even when memory is on
```

Settable via `/config …` (added to `KrillConfig.writableKeys`), plus a
`/memory on|off|status` TUI convenience.

## 3. Alternatives considered

### 3.1 Ship ChromaDB (or another external vector DB) bundled by default — REJECTED

This was the originating question. Rejected for Krill specifically:

- **Krill has no Python runtime.** It is a Swift/SPM/MLX binary (331 Swift files,
  0 Go; the only `.py` files are vendored MLX build deps under
  `.build/checkouts/`). Chroma is a Python library — bundling it means dragging an
  entire Python stack (onnxruntime, hnswlib, sqlite bindings) into a product that
  currently has zero, or running a sidecar service. Either is a large regression
  in "it just `brew install`s and runs."
- **A second model in RAM, taxing everyone.** Chroma's default embedder is another
  model loaded into the user's unified memory, competing with the chat model they
  actually launched Krill for — and on-by-default makes *every* user pay it.
- **Persistent-state support surface.** A DB means schema versioning, migrations
  across `krill update`, corruption recovery, and "my memory vanished after
  upgrade" reports — a permanent new class of bugs for a feature most users
  haven't asked for.
- **Privacy by surprise.** Writing conversations to disk by default, unrequested,
  reads badly when discovered.

We already have a native embedder (`EmbeddingEngine`) and personal memory is
small (low thousands of entries), where brute-force cosine beats standing up an
ANN index. Chroma buys us nothing here and costs the binary's whole shape. (Chroma
remains fine *colony-side*, where a Python runtime already exists — see the
colony's own decisions; this record is about the shipped Krill product.)

### 3.2 On-by-default memory — REJECTED

Simplest UX ("it just remembers"), but violates outcome #2: every user pays disk +
(for semantic) RAM and inherits silent persistence of their conversations. Opt-in
with a one-line toggle keeps the zero-cost default and makes persistence a
conscious choice. Auto-recall is gated a *second* time (`memory_auto_recall`) so
even an opted-in user chooses the injection behaviour.

### 3.3 Only one flavour — REJECTED

- **Semantic only:** forces the embedder + a model download on everyone who wants
  any memory, and isn't human-greppable. Too heavy for "just remember this note."
- **Text only:** misses meaning-based recall ("what did I say about the deploy
  runbook?" won't match "release checklist"). We want the cheap dependency-free
  tier *and* the smart tier, so users pick. They share one JSONL, so `semantic`
  is strictly `text` + a vector cache — minimal extra surface.

### 3.4 A real vector DB as the source of truth (SQLite/`sqlite-vec`, even native) — REJECTED as the *primary* store

Even an embeddable, no-Python store (`sqlite-vec` links into Swift as a C
extension) makes the **vectors** authoritative. That couples the durable record to
a specific embed model and index format — exactly the thing that "no release may
break memory" forbids. Our inversion (plain-text truth + rebuildable vector cache)
keeps the durable artifact model-agnostic and human-readable. `sqlite-vec` remains
a reasonable *optional cache backend* later, but never the source of truth.

### 3.5 ANN index (HNSW) instead of brute-force cosine — REJECTED (for now)

HNSW pays off at hundreds of thousands of vectors. A single user's memory is low
thousands; brute-force cosine over an in-memory `[[Float]]` is *faster* than
building/maintaining an index and adds no dependency or on-disk index format to
version. Revisit only if real usage shows memories at a scale where linear scan
hurts.

### 3.6 Reuse the chat model's hidden states as embeddings — REJECTED

Avoids a second model, but ties memory vectors to whichever chat model is loaded;
recall would change meaning every time the user switches models, and embeddings
wouldn't be comparable across sessions. A dedicated small encoder
(`EmbeddingEngine`) gives stable, model-pinned vectors and runs without a chat
model loaded (its stated design intent).

### 3.7 Default location alternatives — `~/.krill/memory` CHOSEN over the others

| Option | Why not |
|---|---|
| `~/.krill-memory/` (sibling) | Maximally decoupled from `~/.krill`, but a second top-level dotdir is less discoverable and splits "Krill's stuff" across two roots. |
| `~/Documents/Krill Memory/` | Most user-visible, but an unconventional spot for a CLI's data and clutters Documents. |
| **`~/.krill/memory/` (child) — CHOSEN** | Discoverable next to `config.toml`/`models/`; already survives uninstall today (`install.sh` only wipes `/usr/local/libexec/krill`; the formula has no `zap`). Trade-off: it lives *inside* `~/.krill`, so we must guarantee no cleanup path deletes it — addressed by the §5 guard + test. BYOM override covers anyone who wants it elsewhere. |

### 3.8 Tools-only, or auto-recall-only — REJECTED in favour of both, phased

Tools-only misses the seamless "it remembers without being asked" feel;
auto-recall-only hides the mechanism and spends context tokens the user can't
see. Shipping tools first (Phase 1) gives a safe, inspectable, fully-functional
memory; auto-recall (Phase 2) layers the seamless experience behind its own flag.

## 4. Why this over the others

The deciding trade-offs, in order:

1. **Krill's shape is sacred.** No Python, no sidecar, one binary. That alone
   eliminates Chroma and any external DB as the default and points at "reuse the
   embedder we already ship."
2. **Durability is a contract, not a hope.** Inverting truth (text) and cache
   (vectors) is what lets us promise that no release breaks memory — the property
   the user asked for most explicitly.
3. **Zero cost when off.** Opt-in default + MLX kept off the text path means the
   feature is invisible to anyone who doesn't want it.
4. **Proven local pattern.** `SearchBackend` already solved "pluggable backend,
   safe default, config-selected, BYO upgrade" in this codebase; reusing it keeps
   the design legible and the review small.
5. **Right-sized engineering.** Personal-scale data makes brute-force cosine and
   plain JSONL not just adequate but *better* than a DB/ANN — less code, fewer
   deps, nothing to migrate.

## 5. Consequences

- **For users:** opt-in long-term memory in two flavours; portable, inspectable,
  user-owned files; survives updates/reinstalls; no change at all if left off.
- **Durability guard (new requirement this introduces):** because the default dir
  is inside `~/.krill`, every current and future cleanup path (`krill rm`, blob/
  cache prune, a hypothetical `krill uninstall`) must **explicitly skip
  `memory/`**. This record makes that a hard invariant, enforced by a test that
  simulates a prune and asserts `~/.krill/memory` survives (see §6).
- **New config + tools:** five `memory_*` keys, three tools, one `/memory` TUI
  command. Tools are registered only when enabled, so the off-path tool list is
  unchanged.
- **Semantic cost is bounded and lazy:** a small encoder (~hundreds of MB) loads
  only when `memory_backend = semantic`, and `EmbeddingEngine` is built to run
  without disturbing the chat model.
- **Cons we accept:** (a) keyword recall in `text` mode can miss paraphrases — the
  documented fix is "use `semantic`"; (b) brute-force cosine is linear in entry
  count — fine at personal scale, revisited only if that assumption breaks;
  (c) re-embedding after a model/schema change costs a one-time pass — acceptable,
  and strictly better than the data-loss it prevents.
- **First multi-flavour, embedder-backed tool feature** — establishes the
  `MemoryEmbedder` injection seam other features (e.g. local RAG) can reuse.

## 6. Testing / verification

- **Durability guard:** a test that creates `~/.krill/memory/entries.jsonl`, runs
  every cleanup path, and asserts the file still exists — the regression gate for
  the §5 invariant.
- **Cache rebuild:** write `entries.jsonl` + an `index/meta.json` with a mismatched
  `embedModel`/`dim`/`schemaVersion`; assert `SemanticMemoryStore` re-embeds and
  recalls correctly rather than erroring or losing data.
- **Off-path is zero-cost:** assert that with `memory_backend = off` no memory
  tools are registered, no files are created, and nothing in `KrillEngine` is
  loaded.
- **Flavour parity:** the same `entries.jsonl` is readable by both `text` and
  `semantic` stores; switching backends never loses entries.
- **Round-trip recall:** `save` then `search`/`recent` returns the entry; semantic
  recall ranks a paraphrase above an unrelated entry; text recall matches on
  keyword.
- **BYOM:** `KRILL_MEMORY_DIR` and `memory_dir` both redirect reads/writes;
  precedence is env > config > default.
- **Embedder-missing fallback:** with `semantic` selected but no embed model
  present, `search` degrades to keyword and surfaces an actionable note rather
  than failing.
