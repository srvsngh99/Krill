# Design Decisions (ADRs)

This folder is the home for Krill's **design decision records**. Whenever we make
a non-trivial architectural or feature choice - especially one where we picked
one approach over real alternatives - we write it down here so the *why* outlives
the moment.

Each record captures, for one decision:
- **Context** - the problem and what production relied on before.
- **Decision** - what we chose, concretely.
- **Alternatives considered** - the other real options, with examples.
- **Why this over the others** - the trade-offs that decided it.
- **Consequences** - pros, cons, and what it changes (the feature-change detail).
- **Testing/verification** - how we prove it works and does not regress.

This is not API docs or a tutorial; it is the reasoning trail. If you are about
to ask "why is it done this way and not the obvious other way," the answer should
be here.

## Convention

- One file per decision, numbered: `NNNN-short-kebab-title.md` (zero-padded,
  monotonically increasing). Numbers are never reused, even if a record is later
  superseded (mark it superseded, keep the file).
- A record is immutable once adopted, except to add a `Superseded by NNNN` note.
  A changed decision gets a NEW record that references the old one.
- Keep it grounded: cite real files and `file:line`, show concrete examples.

## Template

```markdown
# NNNN. <Title>

Status: proposed | adopted | superseded by NNNN. Date: YYYY-MM-DD. Owner: <name>.

## Context
<the problem; what the system did before; what prompted the change>

## Decision
<what we chose, concretely, with the key code touchpoints>

## Alternatives considered
<each real option, with a short example of how it would behave, and why not>

## Why this over the others
<the deciding trade-offs>

## Consequences
<pros, cons, what changes for users/operators; migration if any>

## Testing / verification
<how it is proven correct and non-regressing>
```

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-tool-calling-and-parser.md) | Tool calling and parsing (hybrid: robust default + per-alias parse override) | adopted | 2026-06-27 |
| [0002](0002-web-search-backends.md) | Web search backends (keyless DuckDuckGo default + BYOK Brave/Tavily; private Kreach behind a build flag) | adopted | 2026-06-28 |
