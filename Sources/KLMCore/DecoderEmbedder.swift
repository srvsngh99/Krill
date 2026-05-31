import MLX

// MARK: - Decoder-LLM embedders

/// Lets a causal-LM backbone (Qwen2/Qwen3, Mistral, Llama) act as a sentence
/// encoder. Decoder-LLM embedders (gte-Qwen2, e5-mistral, SFR) reuse the exact
/// causal forward already validated for chat; the sentence vector comes from
/// pooling the final hidden state (typically last-token, with an EOS appended
/// upstream) and L2-normalizing.
///
/// `lastHiddenState` returns the post-final-norm hidden state `[1, T, H]` from
/// the inner model, *before* the `lm_head` vocab projection. The conformances
/// live here (same module as the models) so they can reach the inner `model`
/// property without widening its access level.

extension QwenForCausalLM: SentenceEmbeddingEncoder {
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        model(tokens, caches: nil)
    }
}

extension MistralForCausalLM: SentenceEmbeddingEncoder {
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        model(tokens, caches: nil)
    }
}

extension LlamaForCausalLM: SentenceEmbeddingEncoder {
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        model(tokens, caches: nil)
    }
}
