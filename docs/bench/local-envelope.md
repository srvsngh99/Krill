# Local inference envelope

Measured with `tools/bench_local_envelope.py`, which drives `krill bench` (forced generation length, warmup pass, averaged runs) on this machine.

Settings: generate 32 tokens, 1 run(s) averaged after 1 warmup. Prompt lengths are exact token counts.

`swap Δ` is the change in system swap across the config — it is the column that separates 'slow' from 'out of envelope'. A negative delta just means macOS reclaimed swap another process had taken.

| model | weights | prompt tok | load s | prefill tok/s | decode tok/s | TTFT ms | peak MB | swap Δ MB | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-0.6b | 0.3 GB | 512 | 0.06 | 4798.5 | 258.8 | 107.0 | 1050.0 | 0 | comfortable |
| qwen3-0.6b | 0.3 GB | 4096 | 0.06 | 2793.3 | 169.2 | 1466.0 | 1630.0 | 0 | comfortable |
| llama-3.2-3b | 1.7 GB | 512 | 0.06 | 949.4 | 98.9 | 539.0 | 2497.0 | 0 | comfortable |
| llama-3.2-3b | 1.7 GB | 4096 | 0.06 | 780.0 | 81.7 | 5251.0 | 3340.0 | -8 | comfortable |
| qwen3-8b | 4.3 GB | 512 | 0.08 | 366.1 | 49.6 | 1398.0 | 5191.0 | 0 | comfortable |
| qwen3-8b | 4.3 GB | 4096 | 0.11 | 306.1 | 40.5 | 13383.0 | 6074.0 | -248 | comfortable |
| gemma-4-12b | 6.7 GB | 512 | 0.14 | 210.0 | 26.9 | 2438.0 | 7474.0 | -16 | comfortable |
| gemma-4-12b | 6.7 GB | 4096 | 0.13 | 183.8 | 24.8 | 22285.0 | 9925.0 | -24 | comfortable |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | 512 | 0.13 | 512.7 | 47.5 | 999.0 | 16980.0 | 3620 | **out of envelope** |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | 4096 | 0.16 | 476.4 | 12.1 | 8598.0 | 18130.0 | 426 | usable |
| muse-glimmer-30b | 18.1 GB | 512 | 0.16 | 71.3 | 7.2 | 7185.0 | 16106.0 | -2042 | usable |
| muse-glimmer-30b | 18.1 GB | 4096 | 0.26 | 78.9 | 6.0 | 51934.0 | 17295.0 | 1846 | usable |
