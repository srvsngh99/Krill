// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KrillLM",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "krillm", targets: ["KLMCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/apple/swift-nio", from: "2.70.0"),
        .package(url: "https://github.com/apple/swift-crypto", from: "3.8.0"),
    ],
    targets: [
        .target(
            name: "KLMCore",
            dependencies: [
                "KLMCache",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KLMCache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KLMTokenizer",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "KLMSampler",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KLMRegistry",
            dependencies: [
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .target(
            name: "KLMServer",
            dependencies: [
                "KLMEngine",
                "KLMRegistry",
                "KLMSampler",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .target(
            name: "KLMEngine",
            dependencies: [
                "KLMCore",
                "KLMCache",
                "KLMTokenizer",
                "KLMSampler",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "KLMCLI",
            dependencies: [
                "KLMEngine",
                "KLMCore",
                "KLMRegistry",
                "KLMServer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .testTarget(
            name: "KLMCoreTests",
            dependencies: ["KLMCore", "KLMCache"]
        ),
        .testTarget(
            name: "KLMRegistryTests",
            dependencies: ["KLMRegistry"]
        ),
    ]
)
