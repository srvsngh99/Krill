import ArgumentParser

@main
struct Krillm: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "krillm",
        abstract: "KrillLM - Fast local LLM inference for Apple Silicon",
        discussion: """
            A Mac-native LLM inference engine built on Apple's MLX framework.
            Runs open-source models locally with optimized Metal GPU acceleration.
            """,
        version: "0.2.0",
        subcommands: [
            RunCommand.self,
            PullCommand.self,
            ServeCommand.self,
            ListCommand.self,
            RemoveCommand.self,
            VersionCommand.self,
        ],
        defaultSubcommand: RunCommand.self
    )
}
