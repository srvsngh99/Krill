# 0002. Web Search Backends

Status: adopted. Date: 2026-06-28. Owner: Sourav. Scope: how `web_search` (and
the `DeepResearch` orchestrator / `POST /research`) reaches the web for users who
install Krill, and how a private local backend is kept out of public releases.

---

## 0. TL;DR

- `web_search` is **backend-agnostic** behind the `SearchBackend` protocol. Adding
  a provider is a new conformer + a `search_backend` value, not a tool change.
- **Public users get web search out of the box.** The default (`search_backend =
  auto`) is the keyless **DuckDuckGo** backend — no API key, no self-hosted infra.
- For reliable, rate-limit-free results, users opt into a **BYOK** backend:
  **Brave** (`brave_api_key`) or **Tavily** (`tavily_api_key`), both with free
  tiers. Self-hosted **SearXNG** remains supported.
- A private **Kreach** backend (a self-hosted crawled index) is **compiled out of
  public release binaries** and present only in local dev builds.

## 1. Problem

Before this change the only backends were SearXNG (self-host) and Kreach (a
private local engine). A fresh `brew install krill` user had no usable backend —
`web_search` returned "not configured". Web search needs to work for downloaded
users without operating any infrastructure.

## 2. Decision

### 2.1 A keyless default + BYOK upgrades

| `search_backend` | Backend | Setup | Notes |
|---|---|---|---|
| `auto` (default), `duckduckgo` | `DuckDuckGoBackend` | none | Keyless. Scrapes `lite.duckduckgo.com`. Best-effort: subject to layout changes / rate limits. |
| `brave` | `BraveBackend` | `brave_api_key` | Brave Search API, free tier. Robust JSON. |
| `tavily` | `TavilyBackend` | `tavily_api_key` | Tavily API, free tier. Agent-oriented. |
| `searxng` | `SearxngBackend` | `searxng_url` | Self-hosted, `json` format enabled. |

API keys are settable via `/config` or `KRILL_BRAVE_API_KEY` / `KRILL_TAVILY_API_KEY`
and are **redacted** in `/config` output (shown as `(set)`/`(none)`, never the value).

Layering rationale: DuckDuckGo makes the tool useful the instant Krill is
installed; users who hit rate limits or want quality drop in a free key. No
Krill-operated search service (no cost/abuse/uptime burden).

### 2.2 Private backend gated by a compile flag

The `KreachBackend`, its `search_backend = "kreach"` wiring, and its `kreach_url`
config surface are guarded by `#if KREACH`. `Package.swift` defines `KREACH` only
when the build sets `KRILL_KREACH=1`:

```sh
make release   # public: KREACH undefined → Kreach absent from the binary
make dev       # local:  KRILL_KREACH=1 → KREACH defined → Kreach compiled in
```

So public releases ship zero Kreach code (verified: no `kreach` symbols in the
`make release` binary), while a local `make dev` build keeps `search_backend =
"kreach"` working. `DeepResearch` / `POST /research` are backend-agnostic and stay
public — they run over whatever backend resolves (DuckDuckGo for public users).

## 3. Consequences

- Web search works for everyone day one; robust with a free key; no service to run.
- The default is a scraped endpoint — inherently best-effort. The BYOK path is the
  documented answer to flakiness, surfaced in the not-configured / error messages.
- Adding `KREACH` is the first compile-flag feature gate in the Swift codebase;
  the env-driven `.define` in `Package.swift` is the pattern for future private
  features.
