import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// `forward(inputsEmbeds:)`, the scatter (`mergedEmbeddings`) and the pooling
/// span, on a tiny synthetic backbone (no weights needed).
final class EmbeddingGemma2MergeTests: XCTestCase {
    private func tinyModel() throws -> EmbeddingGemma2Model {
        let json = """
        {"boi_token_id": 255999, "eoi_token_id": 258882, "image_token_id": 258880,
         "text_config": {"embedding_dim": 12, "head_dim": 8, "hidden_size": 16,
          "hidden_size_per_layer_input": 8, "intermediate_size": 32,
          "layer_types": ["sliding_attention", "full_attention"],
          "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
          "per_layer_config": {"1": {"head_dim": 16, "num_key_value_heads": 1}},
          "sliding_window": 64, "vocab_size": 50}}
        """
        let m = EmbeddingGemma2Model(try JSONDecoder().decode(EmbeddingGemma2Config.self, from: Data(json.utf8)))
        MLXRandom.seed(5)
        let p = m.parameters().flattened().map { (k, v) -> (String, MLXArray) in
            k.hasSuffix("layer_scalar") ? (k, MLXArray([Float(0.5)])) : (k, MLXRandom.normal(v.shape) * 0.1)
        }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.setComputeDtype(.float32)
        eval(m)
        return m
    }

    func testConfigReadsModalityTokens() throws {
        let m = try tinyModel()
        XCTAssertEqual(m.config.modalityTokens.boi, 255999)
        XCTAssertEqual(m.config.modalityTokens.eoi, 258882)
        XCTAssertEqual(m.config.modalityTokens.image, 258880)
        // omitted ids fall back to the checkpoint's
        XCTAssertEqual(m.config.modalityTokens.video, EG2ModalityTokens.checkpointDefaults.video)
        XCTAssertEqual(m.config.modalityTokens.boa, 256000)
    }

    func testInputsEmbedsEntryEqualsTokenEntry() throws {
        let m = try tinyModel()
        let tokens = MLXArray([Int32]([3, 9, 4, 7, 1])).reshaped(1, 5)
        let a = m.forward(tokens, lengths: [5])
        let b = m.forward(inputsEmbeds: m.embedText(tokens), lengths: [5])
        XCTAssertEqual((a - b).abs().max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testScatterReplacesOnlyPlaceholderRowsAndScalesOnlyText() throws {
        let m = try tinyModel()
        // placeholders at 2..4; ids kept inside the tiny vocab
        let seq = EG2Sequence(ids: [2, 7, 9, 9, 9, 8, 1],
                              spans: [EG2SoftSpan(modality: .image, start: 2, count: 3)])
        let soft = MLXRandom.normal([3, 16])
        let e = try m.mergedEmbeddings(seq, features: [.image: [soft]])
        XCTAssertEqual(e.shape, [1, 7, 16])
        let text = m.embedText(MLXArray([Int32]([2, 7, 0, 0, 0, 8, 1])).reshaped(1, 7))
        // text rows (markers included) are the scaled table rows; soft rows are the features, UNscaled
        for t in [0, 1, 5, 6] {
            XCTAssertEqual((e[0, t, 0...] - text[0, t, 0...]).abs().max().item(Float.self), 0,
                           accuracy: 1e-6, "row \(t)")
        }
        XCTAssertEqual((e[0, 2 ..< 5, 0...] - soft).abs().max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testScatterRejectsCountMismatches() throws {
        let m = try tinyModel()
        let seq = EG2Sequence(ids: [2, 9, 9, 1], spans: [EG2SoftSpan(modality: .image, start: 1, count: 2)])
        XCTAssertThrowsError(try m.mergedEmbeddings(seq, features: [:]))
        XCTAssertThrowsError(try m.mergedEmbeddings(seq, features: [.image: [MLXRandom.normal([3, 16])]]))
        XCTAssertThrowsError(try m.mergedEmbeddings(
            seq, features: [.image: [MLXRandom.normal([2, 16]), MLXRandom.normal([2, 16])]]))
        XCTAssertNoThrow(try m.mergedEmbeddings(seq, features: [.image: [MLXRandom.normal([2, 16])]]))
    }

    func testPoolingCoversEveryPositionIncludingSoftTokensAndMarkers() throws {
        let m = try tinyModel()
        let seq = EG2Sequence(ids: [2, 7, 9, 9, 8, 1], spans: [EG2SoftSpan(modality: .image, start: 2, count: 2)])
        let e = try m.mergedEmbeddings(seq, features: [.image: [MLXRandom.normal([2, 16])]])
        let hidden = m.forward(inputsEmbeds: e, lengths: [6]).asType(.float32)  // [1, 6, 12]
        let want = hidden.mean(axis: 1)  // plain mean over ALL 6 positions
        let got = m.pooled(inputsEmbeds: e, lengths: [6])
        XCTAssertEqual((got - want).abs().max().item(Float.self), 0, accuracy: 1e-5)
        // padded: a row padded to 9 pools over its real 6 only
        let padded = concatenated([e, MLXArray.zeros([1, 3, 16])], axis: 1)
        let g2 = m.pooled(inputsEmbeds: padded, lengths: [6])
        XCTAssertEqual((g2 - want).abs().max().item(Float.self), 0, accuracy: 1e-4)
    }
}
