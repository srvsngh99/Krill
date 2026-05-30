/// Structured-output format requested for a generation, threaded from the
/// server through `generate(...)` into the decode loop to drive
/// grammar-constrained decoding (parity plan §8).
///
/// - `.json` (Stage A) constrains the decoder so every emitted token keeps
///   the output a valid prefix of a JSON value.
/// - `.jsonSchema(String)` (Stage B) additionally constrains it to match a
///   JSON Schema (the associated string is the schema JSON): declared keys,
///   required fields, `additionalProperties`, array `items`, scalar types,
///   `enum` / `const`. Unsupported schema constructs relax to "any value"
///   (still valid JSON); if the schema cannot be compiled at all the engine
///   falls back to the `.json` validity mask, and the server's system-prompt
///   guidance + post-extraction `coerce` remain as a final backstop.
/// - `.regex(String)` (Stage C) constrains the output to a full match of a
///   regular expression (the associated string is the pattern): the output
///   must be exactly a string in the regex's language. Unlike the JSON
///   formats the output is NOT JSON, so the server does not JSON-extract it.
///   If the pattern cannot be compiled the engine disables the mask (the
///   request decodes unconstrained) and the system-prompt guidance applies.
public enum OutputFormat: Sendable {
    case json
    case jsonSchema(String)
    case regex(String)
}
