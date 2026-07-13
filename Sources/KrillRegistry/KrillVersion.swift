import Foundation

/// The single source of truth for Krill's version string in Swift
/// code. Bumping a release is therefore a two-line change: write the
/// new version to `VERSION` at the repo root, then update this
/// constant. A `KrillVersionMatchesVersionFileTests` test asserts
/// the two are in sync at build time.
///
/// Lives in `KrillRegistry` because the registry already exposes
/// catalog and discovery primitives that the server and CLI import
/// (`Sources/KrillServer/OllamaCompat.swift` and the CLI's version
/// command both already `import KrillRegistry`). Keeping the constant
/// here avoids pulling a new module into either caller.
public let KrillVersion: String = "0.16.2"

/// Convenience: same string with a leading "v", matching git tag
/// conventions (`v0.6.0`).
public let KrillVersionTag: String = "v" + KrillVersion
