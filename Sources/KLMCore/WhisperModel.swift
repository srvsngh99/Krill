import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Native Whisper runtime (encoder)

/// Swift + MLX port of OpenAI Whisper (the `mlx-examples` / mlx-community
/// `weights.npz` layout is the correctness oracle). Module keys mirror the
/// checkpoint hierarchy exactly (`encoder.*`, `decoder.*`) so weights load
/// with no remapping. See `WhisperMel` for the audio front-end.

/// Whisper dimensions (checkpoint `config.json`, OpenAI field names). All
/// values have published defaults so a missing field never breaks load.
public struct WhisperConfig: Sendable {
    public var nMels = 80
    public var nAudioCtx = 1500
    public var nAudioState = 768
    public var nAudioHead = 12
    public var nAudioLayer = 12
    public var nVocab = 51864
    public var nTextCtx = 448
    public var nTextState = 768
    public var nTextHead = 12
    public var nTextLayer = 12

    public init() {}

    public init(from dict: [String: Any]?) {
        guard let d = dict else { return }
        // OpenAI / mlx field names.
        if let v = d["n_mels"] as? Int { nMels = v }
        if let v = d["n_audio_ctx"] as? Int { nAudioCtx = v }
        if let v = d["n_audio_state"] as? Int { nAudioState = v }
        if let v = d["n_audio_head"] as? Int { nAudioHead = v }
        if let v = d["n_audio_layer"] as? Int { nAudioLayer = v }
        if let v = d["n_vocab"] as? Int { nVocab = v }
        if let v = d["n_text_ctx"] as? Int { nTextCtx = v }
        if let v = d["n_text_state"] as? Int { nTextState = v }
        if let v = d["n_text_head"] as? Int { nTextHead = v }
        if let v = d["n_text_layer"] as? Int { nTextLayer = v }
        // HuggingFace field names (so a raw HF download config loads too).
        if let v = d["num_mel_bins"] as? Int { nMels = v }
        if let v = d["max_source_positions"] as? Int { nAudioCtx = v }
        if let v = d["d_model"] as? Int { nAudioState = v; nTextState = v }
        if let v = d["encoder_attention_heads"] as? Int { nAudioHead = v }
        if let v = d["encoder_layers"] as? Int { nAudioLayer = v }
        if let v = d["vocab_size"] as? Int { nVocab = v }
        if let v = d["max_target_positions"] as? Int { nTextCtx = v }
        if let v = d["decoder_attention_heads"] as? Int { nTextHead = v }
        if let v = d["decoder_layers"] as? Int { nTextLayer = v }
    }
}

/// Fixed sinusoidal position embedding, `[length, channels]` (matches the
/// `mlx-examples` `sinusoids` helper; the encoder positions are computed, not
/// stored in the checkpoint).
func whisperSinusoids(length: Int, channels: Int, maxTimescale: Double = 10_000) -> MLXArray {
    let half = channels / 2
    let logInc = log(maxTimescale) / Double(half - 1)
    var data = [Float](repeating: 0, count: length * channels)
    for t in 0 ..< length {
        for i in 0 ..< half {
            let inv = exp(-logInc * Double(i))
            let scaled = Double(t) * inv
            data[t * channels + i] = Float(sin(scaled))
            data[t * channels + half + i] = Float(cos(scaled))
        }
    }
    return MLXArray(data, [length, channels])
}

/// Whisper multi-head attention. `query`/`value`/`out` carry bias; `key` does
/// not (OpenAI convention). Used as encoder self-attn, decoder masked
/// self-attn, and decoder cross-attn.
final class WhisperAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    let nHead: Int

    init(_ nState: Int, _ nHead: Int) {
        self.nHead = nHead
        _query = ModuleInfo(wrappedValue: Linear(nState, nState, bias: true), key: "query")
        _key = ModuleInfo(wrappedValue: Linear(nState, nState, bias: false), key: "key")
        _value = ModuleInfo(wrappedValue: Linear(nState, nState, bias: true), key: "value")
        _out = ModuleInfo(wrappedValue: Linear(nState, nState, bias: true), key: "out")
    }

    /// Self-attention. Computes `k`/`v` from `x`; when `cache` is supplied
    /// (decode), the new keys/values are appended to it. `mask` is an additive
    /// causal mask `[Lq, Lk]` (nil for the unmasked encoder or single-step
    /// decode). Returns the output and the updated `(k, v)` cache.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil,
                        cache: (MLXArray, MLXArray)? = nil)
        -> (MLXArray, (MLXArray, MLXArray)) {
        let q = query(x)
        var k = key(x)
        var v = value(x)
        if let (pk, pv) = cache {
            k = MLX.concatenated([pk, k], axis: 1)
            v = MLX.concatenated([pv, v], axis: 1)
        }
        return (out(qkv(q, k, v, mask: mask)), (k, v))
    }

    /// Cross-attention to encoder features `xa`. The keys/values depend only on
    /// the (fixed) audio features, so they are computed once and reused via
    /// `cache` across decode steps.
    func cross(_ x: MLXArray, xa: MLXArray, cache: (MLXArray, MLXArray)? = nil)
        -> (MLXArray, (MLXArray, MLXArray)) {
        let q = query(x)
        let k: MLXArray
        let v: MLXArray
        if let (ck, cv) = cache {
            (k, v) = (ck, cv)
        } else {
            k = key(xa)
            v = value(xa)
        }
        return (out(qkv(q, k, v, mask: nil)), (k, v))
    }

    private func qkv(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = q.dim(0), Lq = q.dim(1), D = q.dim(2)
        let Lk = k.dim(1)
        let headDim = D / nHead
        let scale = Float(pow(Double(headDim), -0.25))
        let qh = q.reshaped([B, Lq, nHead, headDim]).transposed(0, 2, 1, 3) * MLXArray(scale)
        let kh = k.reshaped([B, Lk, nHead, headDim]).transposed(0, 2, 3, 1) * MLXArray(scale)
        let vh = v.reshaped([B, Lk, nHead, headDim]).transposed(0, 2, 1, 3)
        var qkScores = qh.matmul(kh)                       // [B, H, Lq, Lk]
        if let m = mask {
            qkScores = qkScores + m[0 ..< Lq, 0 ..< Lk]
        }
        let w = MLX.softmax(qkScores.asType(.float32), axis: -1).asType(q.dtype)
        let o = w.matmul(vh)                               // [B, H, Lq, headDim]
        return o.transposed(0, 2, 1, 3).reshaped([B, Lq, D])
    }
}

/// One residual attention block: masked/unmasked self-attn, optional
/// cross-attn (decoder only), and a GELU MLP, each with a pre-LayerNorm.
final class WhisperResidualBlock: Module {
    @ModuleInfo(key: "attn") var attn: WhisperAttention
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: WhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttnLn: LayerNorm?
    @ModuleInfo(key: "mlp1") var mlp1: Linear
    @ModuleInfo(key: "mlp2") var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(_ nState: Int, _ nHead: Int, crossAttention: Bool) {
        _attn = ModuleInfo(wrappedValue: WhisperAttention(nState, nHead), key: "attn")
        _attnLn = ModuleInfo(wrappedValue: LayerNorm(dimensions: nState), key: "attn_ln")
        if crossAttention {
            _crossAttn = ModuleInfo(
                wrappedValue: WhisperAttention(nState, nHead), key: "cross_attn")
            _crossAttnLn = ModuleInfo(
                wrappedValue: LayerNorm(dimensions: nState), key: "cross_attn_ln")
        }
        _mlp1 = ModuleInfo(wrappedValue: Linear(nState, nState * 4, bias: true), key: "mlp1")
        _mlp2 = ModuleInfo(wrappedValue: Linear(nState * 4, nState, bias: true), key: "mlp2")
        _mlpLn = ModuleInfo(wrappedValue: LayerNorm(dimensions: nState), key: "mlp_ln")
    }

    /// Returns the updated hidden state plus this block's self-attn KV (decode
    /// caching) and a freshly computed cross-attn KV (nil on the encoder path).
    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil,
                        mask: MLXArray? = nil,
                        selfKV: (MLXArray, MLXArray)? = nil,
                        crossKV: (MLXArray, MLXArray)? = nil)
        -> (MLXArray, (MLXArray, MLXArray), (MLXArray, MLXArray)?) {
        var h = x
        let (sa, newSelfKV) = attn(attnLn(h), mask: mask, cache: selfKV)
        h = h + sa
        var newCrossKV: (MLXArray, MLXArray)? = nil
        if let ca = crossAttn, let caLn = crossAttnLn, let audio = xa {
            let (cross, ck) = ca.cross(caLn(h), xa: audio, cache: crossKV)
            h = h + cross
            newCrossKV = ck
        }
        h = h + mlp2(geluExact(mlp1(mlpLn(h))))
        return (h, newSelfKV, newCrossKV)
    }
}

/// Exact (erf-based) GELU, matching `torch.nn.functional.gelu` default.
func geluExact(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1.0 + MLX.erf(x / MLXArray(Float(2.0).squareRoot())))
}

/// Whisper audio encoder. Loaded under checkpoint key `encoder`.
public final class WhisperEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [WhisperResidualBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    let config: WhisperConfig

    public init(_ config: WhisperConfig) {
        self.config = config
        _conv1 = ModuleInfo(
            wrappedValue: Conv1d(
                inputChannels: config.nMels, outputChannels: config.nAudioState,
                kernelSize: 3, padding: 1, bias: true),
            key: "conv1")
        _conv2 = ModuleInfo(
            wrappedValue: Conv1d(
                inputChannels: config.nAudioState, outputChannels: config.nAudioState,
                kernelSize: 3, stride: 2, padding: 1, bias: true),
            key: "conv2")
        _blocks = ModuleInfo(
            wrappedValue: (0 ..< config.nAudioLayer).map { _ in
                WhisperResidualBlock(config.nAudioState, config.nAudioHead, crossAttention: false)
            }, key: "blocks")
        _lnPost = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.nAudioState), key: "ln_post")
    }

    /// `mel`: `[B, nFrames, nMels]` (time-major, channels last; the
    /// `WhisperMel` output expanded with a batch dim). Returns audio features
    /// `[B, nAudioCtx, nAudioState]`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = geluExact(conv1(mel))                      // [B, T, D]
        x = geluExact(conv2(x))                            // [B, T/2, D]
        let L = x.dim(1)
        let pos = whisperSinusoids(length: L, channels: config.nAudioState)
        x = x + pos.asType(x.dtype)
        for block in blocks {
            (x, _, _) = block(x)
        }
        return lnPost(x)
    }
}

/// Per-block decoder KV cache: self-attention keys/values grow with each step;
/// cross-attention keys/values are computed once from the audio features and
/// reused.
public final class WhisperKVCache {
    public var selfKV: [(MLXArray, MLXArray)?]
    public var crossKV: [(MLXArray, MLXArray)?]
    /// Number of tokens already in the self-attention cache (position offset).
    public var offset = 0
    public init(layers: Int) {
        selfKV = Array(repeating: nil, count: layers)
        crossKV = Array(repeating: nil, count: layers)
    }
}

/// Whisper text decoder. Loaded under checkpoint key `decoder`.
public final class WhisperDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [WhisperResidualBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm
    let config: WhisperConfig

    public init(_ config: WhisperConfig) {
        self.config = config
        _tokenEmbedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.nVocab, dimensions: config.nTextState),
            key: "token_embedding")
        _positionalEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([config.nTextCtx, config.nTextState]),
            key: "positional_embedding")
        _blocks = ModuleInfo(
            wrappedValue: (0 ..< config.nTextLayer).map { _ in
                WhisperResidualBlock(config.nTextState, config.nTextHead, crossAttention: true)
            }, key: "blocks")
        _ln = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.nTextState), key: "ln")
    }

    /// Additive causal mask `[L, L]` (0 on/below the diagonal, -inf above).
    private func causalMask(_ L: Int, _ dtype: DType) -> MLXArray {
        let neg = Float(-1e9)
        var data = [Float](repeating: 0, count: L * L)
        for i in 0 ..< L {
            for j in (i + 1) ..< L { data[i * L + j] = neg }
        }
        return MLXArray(data, [L, L]).asType(dtype)
    }

    /// `tokens`: `[B, L]` token ids. `audioFeatures`: encoder output
    /// `[B, nAudioCtx, D]`. `cache` advances in place across calls (decode).
    /// Returns vocabulary logits `[B, L, nVocab]`.
    public func callAsFunction(_ tokens: MLXArray, audioFeatures: MLXArray,
                               cache: WhisperKVCache) -> MLXArray {
        let L = tokens.dim(1)
        let offset = cache.offset
        var x = tokenEmbedding(tokens)
        x = x + positionalEmbedding[offset ..< (offset + L)].asType(x.dtype)

        let mask: MLXArray? = L > 1 ? causalMask(L, x.dtype) : nil
        for i in 0 ..< blocks.count {
            let (h, newSelf, newCross) = blocks[i](
                x, xa: audioFeatures, mask: mask,
                selfKV: cache.selfKV[i], crossKV: cache.crossKV[i])
            x = h
            cache.selfKV[i] = newSelf
            cache.crossKV[i] = newCross
        }
        cache.offset += L
        x = ln(x)
        return tokenEmbedding.asLinear(x)        // tied output projection
    }
}

/// Native Whisper model: audio encoder + text decoder. Loaded from a converted
/// KrillLM model dir (`tools/convert_whisper.py`) with top-level `encoder.*`
/// and `decoder.*` keys.
public final class WhisperModel: Module {
    @ModuleInfo(key: "encoder") var encoder: WhisperEncoder
    @ModuleInfo(key: "decoder") var decoder: WhisperDecoder
    public let config: WhisperConfig

    public init(_ config: WhisperConfig) {
        self.config = config
        _encoder = ModuleInfo(wrappedValue: WhisperEncoder(config), key: "encoder")
        _decoder = ModuleInfo(wrappedValue: WhisperDecoder(config), key: "decoder")
    }

    public func newCache() -> WhisperKVCache { WhisperKVCache(layers: config.nTextLayer) }
}
