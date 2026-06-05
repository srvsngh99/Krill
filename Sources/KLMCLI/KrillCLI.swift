import ArgumentParser
import KLMRegistry

@main
struct Krillm: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "krillm",
        abstract: "KrillLM - Fast local LLM inference for Apple Silicon",
        discussion: """
            A Mac-native LLM inference engine built on Apple's MLX framework.
            Runs open-source models locally with optimized Metal GPU acceleration.
            """,
        version: KrillLMVersion,
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
