import Foundation

/// One model entry in a catalog.
///
/// The fields mirror a hardcoded `AliasMap` alias exactly, so a catalog
/// entry resolves through the pull path identically to a built-in one -
/// the catalog is purely a way to add models without rebuilding the
/// binary.
public struct CatalogEntry: Codable, Sendable, Equatable {
    /// Short model name the user types (e.g. `qwen3-4b`).
    public let alias: String
    /// HuggingFace repo path the weights are pulled from.
    public let repo: String
    /// Model family (drives the loader / runtime path).
    public let family: ModelFamily
    /// Human-readable parameter count (e.g. `4B`).
    public let params: String
    /// Quantization class (e.g. `4bit`).
    public let quant: String
    /// Default context length in tokens.
    public let context: Int

    public init(
        alias: String, repo: String, family: ModelFamily,
        params: String, quant: String, context: Int
    ) {
        self.alias = alias
        self.repo = repo
        self.family = family
        self.params = params
        self.quant = quant
        self.context = context
    }

    /// The `ResolvedModel` an alias resolution returns for this entry.
    public var resolved: ResolvedModel {
        ResolvedModel(
            repo: repo, name: alias, family: family,
            params: params, quant: quant, context: context)
    }
}

/// A versioned catalog of model aliases.
///
/// This is the on-disk and over-the-wire JSON shape. It lets new models
/// be added without rebuilding the binary: a catalog can be edited
/// locally or fetched from a remote URL (`krillm catalog refresh`).
public struct ModelCatalog: Codable, Sendable, Equatable {
    /// Schema version. A store rejects a catalog whose schema it does
    /// not understand rather than silently mis-decoding it.
    public let schemaVersion: Int
    /// When this snapshot was produced (ISO-8601). Informational only;
    /// cache staleness is measured from the local file, not this field.
    public let updated: String?
    /// The model entries.
    public let models: [CatalogEntry]

    /// The schema version this build writes and accepts.
    public static let currentSchemaVersion = 1

    public init(
        schemaVersion: Int = currentSchemaVersion,
        updated: String? = nil,
        models: [CatalogEntry]
    ) {
        self.schemaVersion = schemaVersion
        self.updated = updated
        self.models = models
    }
}

/// Why a catalog operation failed.
public enum CatalogError: Error, CustomStringConvertible, Equatable {
    /// The remote URL returned a non-2xx HTTP status.
    case httpStatus(Int)
    /// The payload was not a decodable `ModelCatalog`.
    case malformed(String)
    /// The catalog declared a `schemaVersion` this build does not support.
    case unsupportedSchema(Int)

    public var description: String {
        switch self {
        case .httpStatus(let code):
            return "catalog fetch failed: HTTP \(code)"
        case .malformed(let detail):
            return "catalog payload is not valid JSON for the catalog schema: \(detail)"
        case .unsupportedSchema(let version):
            return "catalog schemaVersion \(version) is not supported by this build "
                + "(expected \(ModelCatalog.currentSchemaVersion)); upgrade krillm"
        }
    }
}

/// Loads, caches, and refreshes a `ModelCatalog` on local disk.
///
/// The cache lives at `<registry baseDir>/catalog.json`. `AliasMap.resolve`
/// consults it as a fallback after the built-in aliases, so a model that
/// is not compiled into the binary can still be pulled by editing that
/// file or running `krillm catalog refresh --url <url>`.
///
/// The type is a thin, `Sendable` value over a file path; constructing
/// one does no I/O.
public struct ModelCatalogStore: Sendable {
    /// Absolute path of the on-disk catalog cache.
    public let catalogURL: URL

    /// - Parameter baseDir: the registry base directory (e.g. `~/.krillm`).
    public init(baseDir: URL) {
        self.catalogURL = baseDir.appendingPathComponent("catalog.json")
    }

    /// The cached catalog, or `nil` if it is absent, unreadable, or
    /// declares an unsupported schema. A wrong-schema cache is treated
    /// as absent rather than fatal so a stale file never bricks `pull`.
    public func load() -> ModelCatalog? {
        guard let data = try? Data(contentsOf: catalogURL),
              let catalog = try? JSONDecoder().decode(ModelCatalog.self, from: data),
              catalog.schemaVersion == ModelCatalog.currentSchemaVersion
        else {
            return nil
        }
        return catalog
    }

    /// Persist a catalog to the local cache, creating the parent
    /// directory if needed.
    public func save(_ catalog: ModelCatalog) throws {
        try FileManager.default.createDirectory(
            at: catalogURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(catalog).write(to: catalogURL, options: .atomic)
    }

    /// Age of the on-disk cache in seconds, or `nil` if there is no cache.
    public func cacheAge() -> TimeInterval? {
        guard let attrs = try? FileManager.default.attributesOfItem(
                atPath: catalogURL.path),
              let modified = attrs[.modificationDate] as? Date
        else {
            return nil
        }
        return Date().timeIntervalSince(modified)
    }

    /// True iff there is no cache, or the cache is older than `ttl`
    /// seconds. A negative or zero `ttl` makes any existing cache stale.
    public func isStale(ttl: TimeInterval) -> Bool {
        guard let age = cacheAge() else { return true }
        return age > ttl
    }

    /// Resolve a model alias against the cached catalog
    /// (case-insensitive), or `nil` if there is no matching entry.
    public func resolve(_ name: String) -> ResolvedModel? {
        let normalized = name.lowercased().trimmingCharacters(in: .whitespaces)
        return load()?.models
            .first { $0.alias.lowercased() == normalized }?
            .resolved
    }

    /// Fetch a catalog from a remote URL, validate its schema, write it
    /// to the local cache, and return it.
    ///
    /// - Throws: `CatalogError` on a non-2xx status, an undecodable
    ///   payload, or an unsupported schema version. The cache is left
    ///   untouched on any failure.
    public func fetch(
        from url: URL, session: URLSession = .shared
    ) async throws -> ModelCatalog {
        let (data, response) = try await session.data(from: url)
        if let http = response as? HTTPURLResponse,
           !(200...299).contains(http.statusCode) {
            throw CatalogError.httpStatus(http.statusCode)
        }
        let catalog: ModelCatalog
        do {
            catalog = try JSONDecoder().decode(ModelCatalog.self, from: data)
        } catch {
            throw CatalogError.malformed(String(describing: error))
        }
        guard catalog.schemaVersion == ModelCatalog.currentSchemaVersion else {
            throw CatalogError.unsupportedSchema(catalog.schemaVersion)
        }
        try save(catalog)
        return catalog
    }
}
