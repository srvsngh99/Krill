import Foundation
import KrillHarness

/// Bridges the `AgentLoop`'s async `PermissionGate.approve` (called from the
/// background run Task) to a y/n/a prompt rendered on the main TUI task. The
/// loop suspends on a continuation while the render loop shows the pending call
/// and waits for a keystroke to `resolve` it.
///
/// Same cross-task discipline as the code TUI's `EventQueue`: the only shared
/// state is this lock-guarded object; all UI state stays on the main task. The
/// background task touches `approve` (and is resumed by `resolve`); the main
/// task polls `pending` and calls `resolve`.
final class TUIApprover: PermissionGate, @unchecked Sendable {
    struct Request: Equatable { let toolName: String; let argumentsJSON: String }

    private let lock = NSLock()
    private var request: Request?
    private var continuation: CheckedContinuation<Bool, Never>?
    /// Tools the user chose to "always allow" for the rest of the session.
    private var sticky: Set<String> = []

    // MARK: PermissionGate (background task)

    func approve(toolName: String, argumentsJSON: String) async -> Bool {
        // The continuation closure is synchronous, so all locking stays in the
        // sync `register` (NSLock is unavailable from async contexts). A sticky
        // tool resumes immediately; otherwise it parks until the UI resolves it.
        await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            register(toolName: toolName, argumentsJSON: argumentsJSON, cont: cont)
        }
    }

    private func register(
        toolName: String, argumentsJSON: String, cont: CheckedContinuation<Bool, Never>
    ) {
        lock.lock(); defer { lock.unlock() }
        if sticky.contains(toolName) { cont.resume(returning: true); return }
        request = Request(toolName: toolName, argumentsJSON: argumentsJSON)
        continuation = cont
    }

    // MARK: Main task

    /// The tool call awaiting a decision, if any.
    func pending() -> Request? {
        lock.lock(); defer { lock.unlock() }
        return request
    }

    /// Answer the pending prompt. `always` adds the tool to the session
    /// always-allow set so it never prompts again. No-op if nothing is pending.
    func resolve(_ allow: Bool, always: Bool = false) {
        lock.lock()
        guard let cont = continuation, let req = request else { lock.unlock(); return }
        if allow, always { sticky.insert(req.toolName) }
        continuation = nil
        request = nil
        lock.unlock()
        cont.resume(returning: allow)
    }

    /// Drop any always-allow grants (used when a fresh agent conversation starts).
    func reset() {
        lock.lock(); sticky.removeAll(); lock.unlock()
    }
}
