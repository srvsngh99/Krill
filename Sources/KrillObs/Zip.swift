import Foundation

/// Minimal ZIP writer (stored entries, no compression) so `krill bug-report` needs no external
/// `zip` binary and no dependency. Output is a standard ZIP any tool can open.
public struct ZipWriter {
    private var body = Data()
    private var central = Data()
    private var count: UInt16 = 0

    public init() {}

    public mutating func add(name: String, data: Data, date: Date = Date()) {
        let nameBytes = Data(name.utf8)
        let crc = Self.crc32(data)
        let (dosTime, dosDate) = Self.dos(date)
        let offset = UInt32(body.count)
        var local = Data()
        local.le32(0x04034b50); local.le16(20); local.le16(0x0800); local.le16(0)
        local.le16(dosTime); local.le16(dosDate); local.le32(crc)
        local.le32(UInt32(data.count)); local.le32(UInt32(data.count))
        local.le16(UInt16(nameBytes.count)); local.le16(0)
        local.append(nameBytes)
        body.append(local); body.append(data)

        central.le32(0x02014b50); central.le16(20); central.le16(20); central.le16(0x0800); central.le16(0)
        central.le16(dosTime); central.le16(dosDate); central.le32(crc)
        central.le32(UInt32(data.count)); central.le32(UInt32(data.count))
        central.le16(UInt16(nameBytes.count)); central.le16(0); central.le16(0)
        central.le16(0); central.le16(0); central.le32(0); central.le32(offset)
        central.append(nameBytes)
        count += 1
    }

    public func finish() -> Data {
        var out = body
        let start = UInt32(out.count)
        out.append(central)
        out.le32(0x06054b50); out.le16(0); out.le16(0); out.le16(count); out.le16(count)
        out.le32(UInt32(central.count)); out.le32(start); out.le16(0)
        return out
    }

    static func dos(_ d: Date) -> (UInt16, UInt16) {
        var t = time_t(d.timeIntervalSince1970); var tmv = tm(); localtime_r(&t, &tmv)
        let time = UInt16(tmv.tm_hour) << 11 | UInt16(tmv.tm_min) << 5 | UInt16(tmv.tm_sec / 2)
        let year = max(0, Int(tmv.tm_year) - 80)
        let date = UInt16(year) << 9 | UInt16(tmv.tm_mon + 1) << 5 | UInt16(tmv.tm_mday)
        return (time, date)
    }

    static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = c & 1 == 1 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFFFFFF
        for b in data { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFFFFFF
    }
}

extension Data {
    mutating func le16(_ v: UInt16) { append(UInt8(v & 0xFF)); append(UInt8(v >> 8)) }
    mutating func le32(_ v: UInt32) { for s in stride(from: 0, to: 32, by: 8) { append(UInt8((v >> UInt32(s)) & 0xFF)) } }
}
