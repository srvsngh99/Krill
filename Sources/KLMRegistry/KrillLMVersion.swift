import Foundation

/// The single source of truth for KrillLM's version string in Swift
/// code. Bumping a release is therefore a two-line change: write the
/// new version to `VERSION` at the repo root, then update this
/// constant. A `KrillLMVersionMatchesVersionFileTests` test asserts
/// the two are in sync at build time.
///
/// Lives in `KLMRegistry` because the registry already exposes
/// catalog and discovery primitives that the server and CLI import
/// (`Sources/KLMServer/OllamaCompat.swift` and the CLI's version
/// command both already `import KLMRegistry`). Keeping the constant
/// here avoids pulling a new module into either caller.
public let KrillLMVersion: String = "0.4.0"

/// Convenience: same string with a leading "v", matching git tag
/// conventions (`v0.4.0`).
public let KrillLMVersionTag: String = "v" + KrillLMVersion
