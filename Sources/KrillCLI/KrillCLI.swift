import ArgumentParser
import KrillRegistry

@main
struct Krill: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "krill",
        abstract: "Krill - the model and the agent, in one Mac-native binary.",
        discussion: """
            Runs open models locally on Apple's MLX framework (text, vision and \
            voice), serves OpenAI-, Ollama- and Anthropic-compatible APIs, and \
            runs its own coding agent (`krill code`) or lends its engine to other \
            agents (`krill launch`).

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
            UICommand.self,
            ListCommand.self,
            CatalogCommand.self,
            RemoveCommand.self,
            CreateCommand.self,
            ShowCommand.self,
            CpCommand.self,
            StopCommand.self,
            BenchCommand.self,
            PerplexityCommand.self,
            QuantizeCommand.self,
            DebugCommand.self,
            UpdateCommand.self,
            VersionCommand.self,
        ],
        defaultSubcommand: RunCommand.self
    )

}
