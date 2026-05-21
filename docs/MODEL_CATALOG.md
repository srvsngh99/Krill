# Remote Model Catalog

The model catalog lets new models be pulled **without rebuilding the
binary**. KrillLM ships a curated, compiled-in `AliasMap` (the
recommended MLX-quantized models per family); the catalog is an
optional JSON file that supplements it.

## Resolution order

`krillm pull <name>` resolves a name in this order:

1. **Built-in alias** - the curated, tested `AliasMap`.
2. **Catalog entry** - the on-disk catalog cache, when present.
3. **Raw HF repo path** - anything containing `/`.
4. Otherwise: unknown model error.

The built-in map always wins, so a catalog can never silently shadow a
curated alias with a different (or malicious) repo.

## Catalog file

The cache lives at `~/.krillm/catalog.json`
(`krillm catalog path` prints the exact location). Schema:

```json
{
  "schemaVersion": 1,
  "updated": "2026-05-22",
  "models": [
    {
      "alias": "qwen3-next-8b",
      "repo": "mlx-community/Qwen3-Next-8B-4bit",
      "family": "qwen",
      "params": "8B",
      "quant": "4bit",
      "context": 32768
    }
  ]
}
```

`family` must be a known `ModelFamily` raw value (`llama`, `qwen`,
`mistral`, `gemma`, `gemma4`, `phi`, `glm`, `deepseek`, `bert`,
`qwen2_5_vl`, `moe`, `reranker`). A catalog whose `schemaVersion` this
build does not recognize is ignored rather than mis-decoded.

## CLI

```text
krillm catalog list                 # built-in aliases + catalog models
krillm catalog refresh --url <url>  # fetch a remote catalog, cache it
krillm catalog path                 # print the cache file path + age
```

`refresh` also reads the `KRILL_CATALOG_URL` environment variable when
`--url` is omitted. The URL may be `https`, `http`, or `file`. A fetch
that fails (non-2xx status, unparseable payload, unsupported schema)
leaves the existing cache untouched.

## HTTP discovery

`GET /v1/catalog` returns the merged catalog (built-in plus cached
catalog models) so external tooling - managers, bots - can discover
what `pull` accepts without shelling out to the CLI:

```json
{
  "models": [
    { "alias": "llama-3.2-3b", "repo": "...", "family": "llama",
      "params": "3B", "quant": "4bit", "context": 131072,
      "source": "builtin" }
  ],
  "builtin_count": 39,
  "catalog_count": 1
}
```

Each model carries a `source` of `builtin` or `catalog`.

## Staleness

`ModelCatalogStore.isStale(ttl:)` reports whether the cache is older
than a caller-supplied TTL, measured from the file's modification time.
KrillLM does not auto-refresh today; `isStale` is exposed so a manager
or a future scheduled refresh can decide when to call `refresh`.
