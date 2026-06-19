import Foundation
import KrillHarness

/// Terminal `PermissionGate` for `--permission-mode ask`: print the pending
/// tool call and read a one-key answer from stdin. `[a]lways` remembers the
/// tool name for the rest of the run so the user is not re-prompted for it.
///
/// The blocking `readLine()` is bridged off the Swift-concurrency cooperative
/// pool (same pattern as `BashTool`) so it does not stall other async work, and
/// the always-allow set is guarded by a lock because the gate is `Sendable`.
final class StdinApprover: PermissionGate, @unchecked Sendable {
    private let lock = NSLock()
    private var alwaysAllow: Set<String> = []

    func approve(toolName: String, argumentsJSON: String) async -> Bool {
        if isRemembered(toolName) { return true }

        let answer = await readAnswer(
            prompt: "\n  Allow \(toolName)(\(argumentsJSON))? [y]es / [N]o / [a]lways: ")
        switch answer {
        case "y", "yes":
            return true
        case "a", "always":
            remember(toolName)
            return true
        default:
            // Anything else (incl. empty line or EOF on a piped stdin) is a
            // safe "no" - mutating tools never run without an explicit yes.
            return false
        }
    }

    // Synchronous lock helpers: NSLock cannot be held across an await in Swift 6.
    private func isRemembered(_ tool: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return alwaysAllow.contains(tool)
    }

    private func remember(_ tool: String) {
        lock.lock(); defer { lock.unlock() }
        alwaysAllow.insert(tool)
    }

    private func readAnswer(prompt: String) async -> String {
        await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            DispatchQueue.global().async {
                FileHandle.standardOutput.write(Data(prompt.utf8))
                let line = readLine(strippingNewline: true)?
                    .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
                cont.resume(returning: line)
            }
        }
    }
}
