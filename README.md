# KrillLM

A faster, Mac-native LLM inference CLI for Apple Silicon.

Built on Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework via mlx-swift. Ships as a single CLI binary.

## Status

**Phase 1 MVP** - Llama 3 support, `krillm run` command only.

## Build

```bash
swift build -c release --arch arm64
```

The binary lands at `.build/release/krillm`.

## Usage

```bash
# Run a model (provide path to mlx-community model directory)
krillm run /path/to/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

# Single-shot prompt
krillm run /path/to/model "What is the capital of France?"

# Version info
krillm version
```

## Requirements

- macOS 14+
- Apple Silicon (M1 or newer)
- Swift 6.0+
- Model weights in MLX safetensors format (from [mlx-community](https://huggingface.co/mlx-community))

## Architecture

```
KLMCLI           CLI entry point (swift-argument-parser)
KLMEngine        Inference orchestration (prefill + decode loop)
KLMCore          Model definitions (Llama 3, more to come)
KLMCache         KV cache implementations
KLMTokenizer     Tokenizer wrapper (swift-transformers)
KLMSampler       Sampling strategies (greedy, temperature)
```

## License

MIT
