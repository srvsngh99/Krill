import ArgumentParser
import KrillRegistry

@main
struct Krill: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "krill",
        abstract: "Krill - a fast, lean LLM runtime, built for Mac.",
        discussion: """
            A Mac-native LLM runtime on Apple's MLX framework: run open models \
            locally with Metal acceleration - chat, a full-screen TUI with voice, \
            an Ollama/OpenAI-compatible server, and native multimodal.

            Run `krill run <model>` to chat, or set default_model in
            ~/.krill/config.toml and just run `krill`. A Sourav AI Labs project.
            """,
        version: KrillVersion,
        subcommands: [
            RunCommand.self,
            CodeCommand.self,
            PullCommand.self,
            ServeCommand.self,
            LaunchCommand.self,
            ListCommand.self,
            CatalogCommand.self,
            RemoveCommand.self,
            CreateCommand.self,
            ShowCommand.self,
            CpCommand.self,
            StopCommand.self,
            BenchCommand.self,
            QuantizeCommand.self,
            DebugCommand.self,
            UpdateCommand.self,
            VersionCommand.self,
        ],
        defaultSubcommand: RunCommand.self
    )

    /// ArgumentParser's generated entry point runs commands directly, which left no
    /// place to configure logging before a command could emit its first line. Taking
    /// over `main()` lets diagnostics be routed to stderr first (see `KrillLogging`);
    /// the body then reproduces ArgumentParser's own async dispatch.
    static func main() async {
        KrillLogging.bootstrap()
        do {
            var command = try parseAsRoot(nil)
            if var asyncCommand = command as? AsyncParsableCommand {
                try await asyncCommand.run()
            } else {
                try command.run()
            }
        } catch {
            exit(withError: error)
        }
    }
}
