import XCTest
import KLMRuntime

final class MLXMetalRuntimeTests: XCTestCase {
    func testCandidateURLsMatchMLXLoaderOrder() {
        let root = URL(fileURLWithPath: "/tmp/krill-runtime-test")
        let executableDirectory = root.appendingPathComponent("bin")
        let mainBundleURL = root.appendingPathComponent("Main.xctest")
        let bundleResourceURL = root.appendingPathComponent("Main.xctest/Contents/Resources")
        let frameworkResourceURL = root.appendingPathComponent("Cmlx.framework/Resources")

        let context = MLXMetalResourceLocator.Context(
            executableDirectory: executableDirectory,
            mainBundleURL: mainBundleURL,
            bundleResourceURLs: [bundleResourceURL],
            frameworkResourceURLs: [frameworkResourceURL],
            currentDirectory: root
        )

        let paths = MLXMetalResourceLocator.candidateURLs(context: context).map(\.path)

        XCTAssertEqual(paths[0], "/tmp/krill-runtime-test/bin/mlx.metallib")
        XCTAssertEqual(paths[1], "/tmp/krill-runtime-test/bin/Resources/mlx.metallib")
        XCTAssertTrue(paths.contains("/tmp/krill-runtime-test/Main.xctest/mlx-swift_Cmlx.bundle/default.metallib"))
        XCTAssertTrue(paths.contains("/tmp/krill-runtime-test/Main.xctest/Contents/Resources/mlx-swift_Cmlx.bundle/default.metallib"))
        XCTAssertTrue(paths.contains("/tmp/krill-runtime-test/Cmlx.framework/Resources/default.metallib"))
        XCTAssertTrue(paths.contains("/tmp/krill-runtime-test/bin/Resources/default.metallib"))
        XCTAssertEqual(paths.last, "/tmp/krill-runtime-test/default.metallib")
    }

    func testFirstAvailableResourceUsesPreciseSearchPaths() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-runtime-test-\(UUID().uuidString)")
        let executableDirectory = root.appendingPathComponent("bin")
        let bundleDirectory = root
            .appendingPathComponent("Main.xctest/Contents/Resources")
            .appendingPathComponent("\(MLXMetalResourceLocator.swiftPMBundleName).bundle")
        try FileManager.default.createDirectory(
            at: bundleDirectory,
            withIntermediateDirectories: true
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }

        let metallib = bundleDirectory.appendingPathComponent("default.metallib")
        FileManager.default.createFile(atPath: metallib.path, contents: Data())

        let context = MLXMetalResourceLocator.Context(
            executableDirectory: executableDirectory,
            mainBundleURL: root.appendingPathComponent("Main.xctest"),
            bundleResourceURLs: [root.appendingPathComponent("Main.xctest/Contents/Resources")],
            frameworkResourceURLs: [],
            currentDirectory: root
        )

        XCTAssertEqual(
            MLXMetalResourceLocator.firstAvailableResourceURL(context: context)?.path,
            metallib.path
        )
    }
}
