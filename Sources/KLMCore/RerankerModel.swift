import Foundation
import MLX
import MLXNN

/// HF `XLMRobertaForSequenceClassification` / `BertForSequenceClassification`
/// head for cross-encoder rerankers (BGE Reranker, etc.).
///
/// Takes the `[CLS]` hidden state from the backbone, projects through a
/// dense layer with tanh activation, then through a single-output
/// projection to produce one relevance logit per (query, document) pair.
/// `num_labels == 1` for the reranker subtype; multi-label classification
/// heads (e.g. NLI) emit `num_labels > 1` and are out of scope here.
final class RerankerClassificationHead: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numLabels: Int = 1) {
        _dense = ModuleInfo(
            wrappedValue: Linear(hiddenSize, hiddenSize), key: "dense")
        _outProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numLabels), key: "out_proj")
    }

    /// `lastHidden [1, T, H] -> logits [1, numLabels]` using the
    /// `[CLS]` token (position 0).
    func callAsFunction(_ lastHidden: MLXArray) -> MLXArray {
        // [CLS] hidden -> [1, H]
        let cls = lastHidden[0..., 0, 0...]
        // Standard XLMRobertaClassificationHead: tanh activation; no
        // dropout at inference time (dropout layers are no-ops in
        // eval mode anyway).
        let h = MLX.tanh(dense(cls))
        return outProj(h)
    }
}

/// Full reranker model: BERT/XLMRoberta backbone + classification head.
/// Weight prefix in the checkpoint is `bert.` or `roberta.` for the
/// backbone and `classifier.` for the head; the loader strips the
/// backbone prefix exactly like the embedding engine does, so this
/// module sees its sub-modules without any prefix.
public final class RerankerModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: BertEmbeddings
    @ModuleInfo(key: "encoder") var encoder: BertEncoder
    @ModuleInfo(key: "classifier") var classifier: RerankerClassificationHead

    public let config: BertEmbeddingConfig
    public let numLabels: Int

    public init(_ cfg: BertEmbeddingConfig, numLabels: Int = 1) {
        self.config = cfg
        self.numLabels = numLabels
        _embeddings = ModuleInfo(wrappedValue: BertEmbeddings(cfg), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: BertEncoder(cfg), key: "encoder")
        _classifier = ModuleInfo(
            wrappedValue: RerankerClassificationHead(
                hiddenSize: cfg.hiddenSize, numLabels: numLabels),
            key: "classifier")
    }

    /// `tokens [1, T] -> logits [1, numLabels]`. For a single-label
    /// reranker (`numLabels == 1`), call sites typically take the
    /// scalar at `[0, 0]` as the relevance score.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        classifier(encoder(embeddings(tokens)))
    }
}
