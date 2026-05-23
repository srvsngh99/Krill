import XCTest
@testable import KLMAgent

final class HardwareInfoTests: XCTestCase {

    // MARK: - vm_stat / system_profiler parsers

    func testParseVMStatFreeSumsFreeAndInactiveAtPageSize() {
        let output = """
        Mach Virtual Memory Statistics: (page size of 16384 bytes)
        Pages free:                              10000.
        Pages active:                         3000000.
        Pages inactive:                          20000.
        Pages speculative:                        500.
        """
        let bytes = SysctlProbe.parseVMStatFree(output)
        // (10000 + 20000) pages * 16384 bytes/page = 491,520,000
        XCTAssertEqual(bytes, 491_520_000)
    }

    func testParseVMStatFallsBackTo4KPageSizeOnMissingHeader() {
        let output = """
        Mach Virtual Memory Statistics:
        Pages free:                              500.
        Pages inactive:                          500.
        """
        let bytes = SysctlProbe.parseVMStatFree(output)
        // 1000 pages * 4096 = 4_096_000
        XCTAssertEqual(bytes, 4_096_000)
    }

    func testParseGPUCoresExtractsInteger() {
        let output = """
        Graphics/Displays:

            Apple M4 Pro:

              Chipset Model: Apple M4 Pro
              Type: GPU
              Bus: Built-In
              Total Number of Cores: 16
              Vendor: Apple (0x106b)
        """
        XCTAssertEqual(SysctlProbe.parseGPUCores(output), 16)
    }

    func testParseGPUCoresReturnsNilWhenAbsent() {
        XCTAssertNil(SysctlProbe.parseGPUCores("(no displays)"))
    }

    // MARK: - Fit classification table

    func testClassifyFitOn16GBMachineMatchesHeadroomTable() {
        // 16 GB machine, 4-bit dense headroom = model_gb * 1.6
        // Tier thresholds: comfortable < 0.5*R, tight < 0.8*R,
        // risky < 0.95*R, else wontFit.
        let hw = HardwareInfo(
            arch: "arm64", chip: "Apple M4 Pro",
            totalRAMBytes: 16 * 1_073_741_824,
            freeRAMBytes: 8 * 1_073_741_824,
            cpuCores: 12, gpuCores: 16,
            freeDiskBytes: 500 * 1_073_741_824,
            metalAvailable: true)

        // 1 GB model: need 1.6 GB, R/2 = 8 → comfortable
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 1 * 1_073_741_824),
            .comfortable)

        // 6 GB model: need 9.6 GB, in (8, 12.8) → tight
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 6 * 1_073_741_824),
            .tight)

        // 8 GB model: need 12.8 GB, in (12.8, 15.2) - boundary;
        // 12.8 < 15.2 → risky
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 8 * 1_073_741_824),
            .risky)

        // 11 GB model: need 17.6 GB > 15.2 → wontFit
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 11 * 1_073_741_824),
            .wontFit)
    }

    func testClassifyFitOn8GBMachineDowngradesModerateModelsToRisky() {
        let hw = HardwareInfo(
            arch: "arm64", chip: "Apple M1",
            totalRAMBytes: 8 * 1_073_741_824,
            freeRAMBytes: 3 * 1_073_741_824,
            cpuCores: 8, gpuCores: 7,
            freeDiskBytes: 50 * 1_073_741_824,
            metalAvailable: true)

        // 1 GB: need 1.6 GB < 4 GB → comfortable
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 1 * 1_073_741_824),
            .comfortable)
        // 3 GB: need 4.8 GB, in (4, 6.4) → tight
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 3 * 1_073_741_824),
            .tight)
        // 4 GB: need 6.4 GB, in (6.4, 7.6) - 6.4 not strictly < 6.4
        // so falls to risky
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 4 * 1_073_741_824),
            .risky)
        // 40 GB (a 70B 4-bit model) - wontFit
        XCTAssertEqual(
            hw.classifyFit(modelSizeBytes: 40 * 1_073_741_824),
            .wontFit)
    }

    func testClassifyFitReportsWontFitOnZeroRAM() {
        let hw = HardwareInfo(
            arch: "arm64", chip: "?", totalRAMBytes: 0, freeRAMBytes: 0,
            cpuCores: 0, gpuCores: nil, freeDiskBytes: 0,
            metalAvailable: true)
        XCTAssertEqual(hw.classifyFit(modelSizeBytes: 1024), .wontFit)
    }

    // MARK: - FitClassification metadata

    func testFitClassificationSeverityMapping() {
        XCTAssertEqual(FitClassification.comfortable.severity, .info)
        XCTAssertEqual(FitClassification.tight.severity, .warn)
        XCTAssertEqual(FitClassification.risky.severity, .risky)
        XCTAssertEqual(FitClassification.wontFit.severity, .wontFit)
    }
}
