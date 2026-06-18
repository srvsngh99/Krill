// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Krill",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "krill", targets: ["KrillCLI"]),
    ],
    dependencies: [
        // 0.31.4+ required: earlier revisions' QuantizedLinear.init drops the
        // quantization `mode` when allocating params, which breaks loading
        // nvfp4 (4-bit-float) checkpoints. See finding_gemma4_12b_unified.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
        // Jinja is swift-transformers' chat-template engine (transitive). We depend
        // on it directly so KrillTokenizer can render a chat template with extra
        // context (`enable_thinking`) that swift-transformers' fixed-context
        // applyChatTemplate does not expose - this is what lets the engine turn on
        // a reasoning model's thinking channel. Pinned to the major it resolves.
        .package(url: "https://github.com/johnmai-dev/Jinja", from: "1.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/apple/swift-nio", from: "2.70.0"),
        .package(url: "https://github.com/apple/swift-crypto", from: "3.8.0"),
    ],
    targets: [
        // macOS libedit (readline-compatible) for the interactive REPL: line
        // editing, history, and tab completion with no third-party dependency.
        .systemLibrary(name: "CEditLine", path: "Sources/CEditLine"),
        // Pure, dependency-free TUI logic (key decoding, text wrap, slash menu)
        // split out so it is unit-testable without a terminal.
        .target(name: "KrillTUI", dependencies: []),
        .target(
            name: "KrillRuntime",
            dependencies: []
        ),
        .target(
            name: "KrillCore",
            dependencies: [
                "KrillRuntime",
                "KrillCache",
                "KrillKernels",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KrillCache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KrillTokenizer",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Jinja", package: "Jinja"),
            ]
        ),
        .target(
            name: "KrillSampler",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KrillGrammar",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KrillKernels",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            exclude: ["FusedSwiGLU.metal"]
        ),
        .target(
            name: "KrillRegistry",
            dependencies: [
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .target(
            name: "KrillServer",
            dependencies: [
                "KrillEngine",
                "KrillCache",
                "KrillRegistry",
                "KrillSampler",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .target(
            name: "KrillEngine",
            dependencies: [
                "KrillCore",
                "KrillCache",
                "KrillTokenizer",
                "KrillSampler",
                "KrillGrammar",
                "KrillRegistry",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KrillAgent",
            dependencies: [
                "KrillRegistry",
            ]
        ),
        .executableTarget(
            name: "KrillCLI",
            dependencies: [
                "KrillEngine",
                "KrillCore",
                "KrillCache",
                "KrillSampler",
                "KrillRegistry",
                "KrillServer",
                "CEditLine",
                "KrillTUI",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .testTarget(
            name: "KrillCoreTests",
            dependencies: [
                "KrillCore",
                "KrillCache",
                "KrillRuntime",
                "KrillRegistry",
                "KrillKernels",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "KrillEngineTests",
            dependencies: [
                "KrillEngine",
                "KrillCore",
                "KrillCache",
                "KrillSampler",
                "KrillRuntime",
                "KrillTokenizer",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "KrillServerTests",
            dependencies: [
                "KrillServer",
                "KrillEngine",
                "KrillRegistry",
                "KrillSampler",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "KrillTUITests",
            dependencies: ["KrillTUI"]
        ),
        .testTarget(
            name: "KrillRegistryTests",
            dependencies: [
                "KrillRegistry",
                .product(name: "Crypto", package: "swift-crypto"),
            ]
        ),
        .testTarget(
            name: "KrillAgentTests",
            dependencies: [
                "KrillAgent",
                "KrillRegistry",
            ]
        ),
        .testTarget(
            name: "KrillGrammarTests",
            dependencies: [
                "KrillGrammar",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "KrillTokenizerTests",
            dependencies: [
                "KrillTokenizer",
            ]
        ),
    ]
)
