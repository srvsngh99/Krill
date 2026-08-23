import Foundation

/// Thread-safe, live permission state for one agent run.
public final class PermissionBox: @unchecked Sendable {
    /// The posture chosen when the run was created. Adaptive remains its visible
    /// identity after its effective policy enters the execution phase.
    public let origin: PermissionMode

    private let lock = NSLock()
    private var storedPolicy: PermissionPolicy

    public init(
        mode: PermissionMode,
        allow: Set<String> = [],
        deny: Set<String> = []
    ) {
        origin = mode
        storedPolicy = PermissionPolicy(
            mode: mode.initialEffective, allow: allow, deny: deny)
    }

    public convenience init(policy: PermissionPolicy) {
        self.init(mode: policy.mode, allow: policy.allow, deny: policy.deny)
    }

    /// Advanced initializer for restoring an origin and its current effective
    /// policy independently (for example, an adaptive remote session reconnect).
    public init(origin: PermissionMode, policy: PermissionPolicy) {
        self.origin = origin
        self.storedPolicy = policy
    }

    public var policy: PermissionPolicy {
        lock.withLock { storedPolicy }
    }

    public var effective: PermissionMode {
        lock.withLock { storedPolicy.mode }
    }

    public var isPlanning: Bool {
        lock.withLock { storedPolicy.mode == .plan || storedPolicy.mode == .adaptive }
    }

    public var chipLabel: String {
        lock.withLock {
            if origin == .adaptive {
                let planning = storedPolicy.mode == .plan || storedPolicy.mode == .adaptive
                return "adaptive (\(planning ? "planning" : "executing"))"
            }
            return storedPolicy.mode.label
        }
    }

    /// Model-facing transition. It can only leave a planning posture and can
    /// never grant unattended execution (`acceptAll`).
    @discardableResult
    public func promote(to mode: PermissionMode) -> Bool {
        lock.withLock {
            guard storedPolicy.mode == .plan || storedPolicy.mode == .adaptive else {
                return false
            }
            guard mode == .ask || mode == .acceptEdits else { return false }
            storedPolicy = PermissionPolicy(
                mode: mode, allow: storedPolicy.allow, deny: storedPolicy.deny)
            return true
        }
    }

    /// Human-facing transition. Humans may tighten or loosen the posture in any
    /// direction; explicit allow/deny lists remain authoritative.
    public func setPolicy(mode: PermissionMode) {
        lock.withLock {
            storedPolicy = PermissionPolicy(
                mode: mode.initialEffective,
                allow: storedPolicy.allow,
                deny: storedPolicy.deny)
        }
    }
}
