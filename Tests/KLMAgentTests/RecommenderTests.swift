import XCTest
import KLMRegistry
@testable import KLMAgent

final class RecommenderTests: XCTestCase {

    // MARK: - Size estimator

    func testEstimatedSize4BitDenseMatchesCheckpointOrderOfMagnitude() {
        // Llama-3.2-1B-4bit is ~700 MB on disk; the estimator should
        // land within ~25 % of that.
        let bytes = Recommender.estimatedSize(params: "1B", quant: "4bit")
        let gb = Double(bytes) / 1_073_741_824.0
        XCTAssertGreaterThan(gb, 0.5)
        XCTAssertLessThan(gb, 1.2)
    }

    func testEstimatedSizeMoEParamsCountAllExperts() {
        // 8x7B Mixtral 4-bit is ~24 GB on disk, NOT 4 GB.
        let bytes = Recommender.estimatedSize(params: "8x7B", quant: "4bit")
        let gb = Double(bytes) / 1_073_741_824.0
        XCTAssertGreaterThan(gb, 20)
    }

    func testEstimatedSizeHandlesQwen3MoELabel() {
        // qwen3-30b-a3b ships params = "30B-A3B": 30 B total, 3 B
        // active. On-disk footprint is the larger value (the full
        // expert set lives on disk). At 4-bit that's ~16-17 GB.
        let bytes = Recommender.estimatedSize(params: "30B-A3B", quant: "4bit")
        let gb = Double(bytes) / 1_073_741_824.0
        XCTAssertGreaterThan(gb, 14, "got \(gb) GB")
        XCTAssertLessThan(gb, 20, "got \(gb) GB")
    }

    func testEstimatedSizeHandlesOLMoELabel() {
        // olmoe-1b-7b ships params = "1B-7B" (active-total ordering).
        // On-disk is the larger value (7 B total); active 1 B is a
        // runtime concern. At 4-bit ~4 GB.
        let bytes = Recommender.estimatedSize(params: "1B-7B", quant: "4bit")
        let gb = Double(bytes) / 1_073_741_824.0
        XCTAssertGreaterThan(gb, 3, "got \(gb) GB")
        XCTAssertLessThan(gb, 5, "got \(gb) GB")
    }

    func testRecommendFiltersOutQwen3MoEOnSmallHardware() {
        // Regression for the Codex round-1 finding: before the
        // composite-label fix, qwen3-30b-a3b sized as 200 MB and
        // recommended as "comfortable" on an 8 GB Mac. Now it
        // should size as ~16 GB and be filtered as wontFit.
        let catalog: [CatalogEntry] = [
            .init(alias: "qwen3-30b-a3b",
                  repo: "x/qwen3moe", family: .moe,
                  params: "30B-A3B", quant: "4bit", context: 32768),
            .init(alias: "olmoe-1b-7b",
                  repo: "x/olmoe", family: .moe,
                  params: "1B-7B", quant: "4bit", context: 4096),
        ]
        let hw = fixtureHardware(ramGB: 8)
        let ranked = Recommender.recommend(
            required: [.textGeneration],
            catalog: catalog,
            hardware: hw)
        // qwen3-30b-a3b sizes to ~16 GB -> wontFit -> dropped.
        // olmoe-1b-7b sizes to ~4 GB -> risky tier on 8 GB -> kept
        // (still shortlistable with a warning, by design).
        // Before this fix qwen3-30b-a3b appeared as "comfortable".
        XCTAssertFalse(ranked.contains { $0.alias == "qwen3-30b-a3b" })
        XCTAssertTrue(ranked.contains { $0.alias == "olmoe-1b-7b" })
        let olmoe = ranked.first { $0.alias == "olmoe-1b-7b" }
        // ~3.8 GB on-disk * 1.6 headroom -> 6.05 GB. On 8 GB, that
        // is just under R*0.8 (6.4 GB) so it lands as `tight`.
        XCTAssertEqual(olmoe?.fit, .tight)
    }

    func testEstimatedSizeFP32MultiplesBytesPerParam() {
        // 23 M params at fp32 should be ~92 MB.
        let bytes = Recommender.estimatedSize(params: "23M", quant: "fp32")
        let mb = Double(bytes) / (1024 * 1024)
        XCTAssertGreaterThan(mb, 80)
        XCTAssertLessThan(mb, 300)  // includes 200 MB overhead
    }

    // MARK: - Ranking

    private func fixtureHardware(ramGB: Double, arch: String = "arm64") -> HardwareInfo {
        HardwareInfo(
            arch: arch, chip: "fixture",
            totalRAMBytes: UInt64(ramGB * 1_073_741_824),
            freeRAMBytes: UInt64(ramGB * 0.5 * 1_073_741_824),
            cpuCores: 8, gpuCores: 8,
            freeDiskBytes: 200 * 1_073_741_824,
            metalAvailable: arch == "arm64")
    }

    func testRecommendForTextOnly16GBPrefersMidsizeProductionNative() {
        let catalog: [CatalogEntry] = [
            .init(alias: "llama-3.2-1b",
                  repo: "x/llama-1b", family: .llama,
                  params: "1B", quant: "4bit", context: 8192),
            .init(alias: "llama-3.2-3b",
                  repo: "x/llama-3b", family: .llama,
                  params: "3B", quant: "4bit", context: 8192),
            .init(alias: "llama-3.1-8b",
                  repo: "x/llama-8b", family: .llama,
                  params: "8B", quant: "4bit", context: 8192),
            .init(alias: "mixtral-8x7b",
                  repo: "x/mixtral", family: .moe,
                  params: "8x7B", quant: "4bit", context: 32768),
        ]
        let hw = fixtureHardware(ramGB: 16)
        let ranked = Recommender.recommend(
            required: [.textGeneration],
            catalog: catalog,
            hardware: hw)
        XCTAssertFalse(ranked.isEmpty)
        // The 8x7B MoE won't fit on 16 GB — should be filtered.
        XCTAssertFalse(ranked.contains { $0.alias == "mixtral-8x7b" })
        // 1B is below the [1.5, 14] sweet-spot band, so it should rank
        // below the 3B and 8B (both production-native, both comfortable,
        // both in the bonus band).
        let topTwo = Array(ranked.prefix(2)).map(\.alias)
        XCTAssertTrue(topTwo.contains("llama-3.2-3b"))
        XCTAssertTrue(topTwo.contains("llama-3.1-8b"))
        XCTAssertFalse(topTwo.contains("llama-3.2-1b"))
    }

    func testRecommendForAudioInputReturnsOnlyGemma4Family() {
        let catalog: [CatalogEntry] = [
            .init(alias: "qwen2.5-1.5b",
                  repo: "x/qwen", family: .qwen,
                  params: "1.5B", quant: "4bit", context: 8192),
            .init(alias: "gemma-4-e2b",
                  repo: "x/g4-e2b", family: .gemma4,
                  params: "2B", quant: "4bit", context: 8192),
            .init(alias: "gemma-4-e4b",
                  repo: "x/g4-e4b", family: .gemma4,
                  params: "4B", quant: "4bit", context: 8192),
        ]
        let hw = fixtureHardware(ramGB: 32)
        let ranked = Recommender.recommend(
            required: [.audioInput],
            catalog: catalog,
            hardware: hw)
        XCTAssertEqual(Set(ranked.map(\.family)), [.gemma4])
        XCTAssertEqual(ranked.count, 2)
    }

    func testRecommendOnIntelMacFiltersOutAppleSiliconOnlyFamilies() {
        let catalog: [CatalogEntry] = [
            .init(alias: "qwen2.5-vl-3b",
                  repo: "x/vl-3b", family: .qwen25vl,
                  params: "3B", quant: "4bit", context: 32768),
            .init(alias: "gemma-4-e2b",
                  repo: "x/g4-e2b", family: .gemma4,
                  params: "2B", quant: "4bit", context: 8192),
            .init(alias: "qwen2.5-3b",
                  repo: "x/q-3b", family: .qwen,
                  params: "3B", quant: "4bit", context: 8192),
        ]
        let hw = fixtureHardware(ramGB: 32, arch: "x86_64")
        let ranked = Recommender.recommend(
            required: [.textGeneration],
            catalog: catalog,
            hardware: hw)
        XCTAssertEqual(ranked.map(\.alias), ["qwen2.5-3b"])
    }

    func testRecommendShortlistRespectsLimit() {
        let catalog: [CatalogEntry] = (0 ..< 10).map { i in
            .init(alias: "model-\(i)",
                  repo: "x/m-\(i)", family: .llama,
                  params: "1B", quant: "4bit", context: 8192)
        }
        let hw = fixtureHardware(ramGB: 32)
        let ranked = Recommender.recommend(
            required: [.textGeneration],
            catalog: catalog,
            hardware: hw,
            limit: 3)
        XCTAssertEqual(ranked.count, 3)
    }

    func testRecommendBuiltInAliasMapCarriesEveryFamily() {
        // Sanity: pulling the built-in alias map into the recommender
        // should give a non-empty shortlist on every common request.
        let entries = CatalogEntry.fromAliasMap()
        XCTAssertGreaterThan(entries.count, 10)
        let hw = fixtureHardware(ramGB: 16)

        let chat = Recommender.recommend(
            required: [.textGeneration], catalog: entries, hardware: hw)
        XCTAssertFalse(chat.isEmpty)

        let tools = Recommender.recommend(
            required: [.tools], catalog: entries, hardware: hw)
        XCTAssertFalse(tools.isEmpty)
    }
}
