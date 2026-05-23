import Foundation

/// `hardware_info` operator tool.
///
/// Returns the user's `HardwareInfo` snapshot as a compact JSON string
/// the router model can read. Captures once per agent session by
/// default - the system prompt already includes the snapshot, but the
/// model may call this explicitly when the user asks "what's my
/// machine?" or after a re-probe is appropriate.
public struct HardwareInfoTool: OperatorTool {
    public let name = "hardware_info"
    public let description =
        "Report the user's machine specs (arch, chip, total RAM, " +
        "free RAM, free disk, GPU cores, Metal availability)."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let snapshot: @Sendable () -> HardwareInfo

    public init(snapshot: @escaping @Sendable () -> HardwareInfo = HardwareInfo.current) {
        self.snapshot = snapshot
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        let hw = snapshot()
        let payload: [String: Any] = [
            "arch": hw.arch,
            "chip": hw.chip,
            "total_ram_gb": round1(hw.totalRAMGB),
            "free_ram_gb": round1(hw.freeRAMGB),
            "free_disk_gb": round1(hw.freeDiskGB),
            "cpu_cores": hw.cpuCores,
            "gpu_cores": hw.gpuCores as Any? ?? NSNull(),
            "metal_available": hw.metalAvailable,
        ]
        return encode(payload)
    }
}

internal func round1(_ x: Double) -> Double {
    (x * 10).rounded() / 10
}

internal func encode(_ obj: Any) -> String {
    guard let data = try? JSONSerialization.data(
        withJSONObject: obj, options: [.sortedKeys]),
          let s = String(data: data, encoding: .utf8)
    else {
        return "{}"
    }
    return s
}
