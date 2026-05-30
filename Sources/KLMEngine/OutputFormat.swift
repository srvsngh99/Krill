/// Structured-output format requested for a generation, threaded from the
/// server through `generate(...)` into the decode loop to drive
/// grammar-constrained decoding (parity plan §8, Stage A).
///
/// `.json` constrains the decoder so every emitted token keeps the output
/// a valid prefix of a JSON value (via `KLMGrammar` / `JSONTokenMask`).
/// JSON-schema requests map to `.json` for now — the value is guaranteed
/// well-formed JSON by the mask, and the schema shape is still enforced by
/// the server's system-prompt guidance + post-extraction `coerce` until a
/// schema→grammar compiler lands (Stage B).
public enum OutputFormat: Sendable {
    case json
}
