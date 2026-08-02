import Foundation

/// Pure security-policy helpers shared by the CLI guard and HTTP handler.
public enum ServerSecurity {
    public static func normalizedAPIKey(_ raw: String?) -> String? {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty,
              !value.contains(where: { $0.isWhitespace }) else { return nil }
        return value
    }

    /// True only for host spellings that bind exclusively to the local machine.
    public static func isLoopbackHost(_ rawHost: String) -> Bool {
        var host = rawHost.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if host == "localhost" || host == "localhost." {
            return true
        }
        if host.hasPrefix("["), host.hasSuffix("]") {
            host = String(host.dropFirst().dropLast())
        }
        if isIPv6Loopback(host) { return true }
        let octets = host.split(separator: ".", omittingEmptySubsequences: false)
        guard octets.count == 4,
              let first = Int(octets[0]), first == 127,
              octets.allSatisfy({ part in
                  guard let value = Int(part) else { return false }
                  return value >= 0 && value <= 255
              }) else {
            return false
        }
        return true
    }

    /// Parse compressed or fully expanded IPv6 and accept only `::1`.
    private static func isIPv6Loopback(_ host: String) -> Bool {
        guard host.contains(":") else { return false }
        let pieces = host.components(separatedBy: "::")
        guard pieces.count <= 2 else { return false }

        func groups(_ side: String) -> [Substring] {
            side.isEmpty ? [] : side.split(separator: ":", omittingEmptySubsequences: false)
        }
        let left = groups(pieces[0])
        let right = pieces.count == 2 ? groups(pieces[1]) : []
        let explicitCount = left.count + right.count
        let zeroFill: Int
        if pieces.count == 2 {
            guard explicitCount < 8 else { return false }
            zeroFill = 8 - explicitCount
        } else {
            guard left.count == 8 else { return false }
            zeroFill = 0
        }
        let explicit = left + right
        guard explicit.allSatisfy({ group in
            !group.isEmpty && group.count <= 4 && UInt16(group, radix: 16) != nil
        }) else { return false }
        let expanded = left.map(String.init)
            + Array(repeating: "0", count: zeroFill)
            + right.map(String.init)
        guard expanded.count == 8 else { return false }
        return expanded.dropLast().allSatisfy { UInt16($0, radix: 16) == 0 }
            && UInt16(expanded[7], radix: 16) == 1
    }

    /// Validate an RFC 6750-style Authorization header against the configured
    /// token. Comparison examines every byte to avoid content-based early exits.
    public static func isAuthorized(authorization: String?, apiKey: String?) -> Bool {
        guard apiKey != nil else { return true }
        guard let expected = normalizedAPIKey(apiKey) else { return false }
        guard let authorization else { return false }
        let fields = authorization.split(whereSeparator: { $0.isWhitespace })
        guard fields.count == 2, fields[0].lowercased() == "bearer" else { return false }
        return constantTimeEqual(String(fields[1]), expected)
    }

    /// Remote unauthenticated binds require an explicit acknowledgement.
    public static func permitsBinding(
        host: String,
        apiKey: String?,
        allowRemoteUnauthenticated: Bool
    ) -> Bool {
        isLoopbackHost(host)
            || normalizedAPIKey(apiKey) != nil
            || allowRemoteUnauthenticated
    }

    private static func constantTimeEqual(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs.utf8)
        let right = Array(rhs.utf8)
        let count = max(left.count, right.count)
        var difference = left.count ^ right.count
        for index in 0..<count {
            difference |= Int(index < left.count ? left[index] : 0)
                ^ Int(index < right.count ? right[index] : 0)
        }
        return difference == 0
    }
}
