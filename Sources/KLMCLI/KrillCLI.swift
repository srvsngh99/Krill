import ArgumentParser
import KLMRegistry

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
            VersionCommand.self,
        ],
        defaultSubcommand: RunCommand.self
    )
}
