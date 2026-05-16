import XCTest
@testable import KLMCore

final class MLXMemoryConfigTests: XCTestCase {

    func testDefaultWhenUnset() {
        let mb = MLXMemoryConfig.resolveCacheLimitMB(environment: [:])
        XCTAssertEqual(mb, MLXMemoryConfig.defaultCacheLimitMB)
    }

    func testExplicitPositiveValue() {
        let mb = MLXMemoryConfig.resolveCacheLimitMB(
            environment: [MLXMemoryConfig.envVar: "512"])
        XCTAssertEqual(mb, 512)
    }

    func testZeroDisablesCap() {
        let mb = MLXMemoryConfig.resolveCacheLimitMB(
            environment: [MLXMemoryConfig.envVar: "0"])
        XCTAssertNil(mb, "0 must disable the cap (legacy unbounded behavior)")
    }

    func testWhitespaceTrimmed() {
        let mb = MLXMemoryConfig.resolveCacheLimitMB(
            environment: [MLXMemoryConfig.envVar: "  128 "])
        XCTAssertEqual(mb, 128)
    }

    func testInvalidFallsBackToDefault() {
        for bad in ["abc", "-1", "1.5", ""] {
            let mb = MLXMemoryConfig.resolveCacheLimitMB(
                environment: [MLXMemoryConfig.envVar: bad])
            XCTAssertEqual(mb, MLXMemoryConfig.defaultCacheLimitMB,
                           "\(bad.isEmpty ? "empty" : bad) must fall back to default")
        }
    }
}
