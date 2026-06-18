import ArgumentParser
import Foundation
import KrillRegistry

/// `krill catalog` - inspect and refresh the model catalog.
///
/// The catalog is a JSON file (`~/.krill/catalog.json`) of model
/// aliases that supplements the curated, compiled-in `AliasMap`. It
/// lets new models be pulled without rebuilding the binary: `pull`
/// resolves a name against the built-in map first, then the catalog.
struct CatalogCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "catalog",
        abstract: "Inspect and refresh the model catalog",
        subcommands: [CatalogList.self, CatalogRefresh.self, CatalogPath.self],
        defaultSubcommand: CatalogList.self
    )
}

/// Format a duration as a compact age string (`45s`, `12m`, `3h`, `2d`).
private func formatAge(_ seconds: TimeInterval) -> String {
    if seconds < 60 { return "\(Int(seconds))s" }
    if seconds < 3600 { return "\(Int(seconds / 60))m" }
    if seconds < 86400 { return "\(Int(seconds / 3600))h" }
    return "\(Int(seconds / 86400))d"
}

private func printRow(_ alias: String, _ params: String, _ family: String) {
    print("  "
        + alias.padding(toLength: 24, withPad: " ", startingAt: 0)
        + params.padding(toLength: 6, withPad: " ", startingAt: 0)
        + family)
}

/// `krill catalog list` - show built-in aliases and catalog entries.
struct CatalogList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List built-in aliases and catalog models"
    )

    func run() throws {
        let store = ModelCatalogStore(baseDir: Registry().baseDir)

        let builtIn = AliasMap.allAliases
        print("Built-in aliases (\(builtIn.count)):")
        for (name, info) in builtIn {
            printRow(name, info.params, info.family.rawValue)
        }

        guard let catalog = store.load() else {
            print("\nNo catalog cached. Add one with: krill catalog refresh --url <url>")
            return
        }
        guard !catalog.models.isEmpty else {
            print("\nCatalog cached but empty (\(store.catalogURL.path)).")
            return
        }
        let ageNote = store.cacheAge().map { ", cached \(formatAge($0)) ago" } ?? ""
        print("\nCatalog models (\(catalog.models.count)\(ageNote)):")
        for entry in catalog.models.sorted(by: { $0.alias < $1.alias }) {
            printRow(entry.alias, entry.params, entry.family.rawValue)
        }
    }
}

/// `krill catalog refresh` - fetch a catalog from a remote URL.
struct CatalogRefresh: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "refresh",
        abstract: "Fetch the model catalog from a remote URL and cache it"
    )

    @Option(name: .long, help: "Catalog URL (overrides the KRILL_CATALOG_URL environment variable)")
    var url: String?

    func run() async throws {
        let source = url ?? ProcessInfo.processInfo.environment["KRILL_CATALOG_URL"]
        guard let source, !source.isEmpty else {
            print("Error: no catalog URL. Pass --url <url> or set KRILL_CATALOG_URL.")
            throw ExitCode.failure
        }
        guard let remoteURL = URL(string: source),
              let scheme = remoteURL.scheme?.lowercased(),
              ["https", "http", "file"].contains(scheme) else {
            print("Error: '\(source)' is not a valid https / http / file URL.")
            throw ExitCode.failure
        }

        let store = ModelCatalogStore(baseDir: Registry().baseDir)
        print("Fetching catalog from \(remoteURL.absoluteString)...")
        do {
            let catalog = try await store.fetch(from: remoteURL)
            print("Cached \(catalog.models.count) model(s) to \(store.catalogURL.path)")
            if let updated = catalog.updated {
                print("Catalog snapshot: \(updated)")
            }
        } catch let error as CatalogError {
            print("Error: \(error.description)")
            throw ExitCode.failure
        } catch {
            // Transport failures (unreachable host, DNS, a missing
            // file:// path) surface as URLError / CocoaError, not
            // CatalogError; keep the curated single-line output.
            print("Error: could not fetch catalog: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}

/// `krill catalog path` - print the catalog cache location.
struct CatalogPath: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "path",
        abstract: "Print the catalog cache file path"
    )

    func run() throws {
        let store = ModelCatalogStore(baseDir: Registry().baseDir)
        print(store.catalogURL.path)
        if let age = store.cacheAge() {
            print("cached \(formatAge(age)) ago")
        } else {
            print("(not cached)")
        }
    }
}
