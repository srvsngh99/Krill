import ArgumentParser
import Foundation

struct VersionCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "version",
        abstract: "Print version and system information"
    )

    func run() throws {
        print("KrillLM v0.2.0")
        print("By Sourav Singh / Sourav AI Labs")
        print("https://github.com/srvsngh99/KrillLM")
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
        return String(cString: brand)
    }
    return "Apple Silicon"
}
