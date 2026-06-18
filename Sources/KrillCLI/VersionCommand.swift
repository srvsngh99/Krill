import ArgumentParser
import Foundation
import KrillRegistry

struct VersionCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "version",
        abstract: "Print version and system information"
    )

    func run() throws {
        print("Krill \(KrillVersionTag)")
        print("By Sourav Singh / Sourav AI Labs")
        print("https://github.com/srvsngh99/Krill")
        print()
        print("Backend: mlx-swift (Apple MLX)")
        print("Platform: \(platform())")
        print("Chip: \(chipInfo())")
    }
}

private func platform() -> String {
    var sysinfo = utsname()
    uname(&sysinfo)
    let machine = withUnsafePointer(to: &sysinfo.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 256) {
            String(cString: $0)
        }
    }
    let release = withUnsafePointer(to: &sysinfo.release) {
        $0.withMemoryRebound(to: CChar.self, capacity: 256) {
            String(cString: $0)
        }
    }
    return "macOS (Darwin \(release), \(machine))"
}

private func chipInfo() -> String {
    var size: Int = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    if size > 0 {
        var brand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let length = brand.firstIndex(of: 0) ?? brand.count
        return String(decoding: brand.prefix(length).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
    return "Apple Silicon"
}
