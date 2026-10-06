import Foundation

/// A SentencePiece-style byte-fallback BPE encoder that starts from Unicode
/// SCALARS (code points), exactly like HuggingFace `tokenizers`.
///
/// Why this exists: swift-transformers' `BPETokenizer.bpe` seeds the merge
/// loop with `Array(token).map(String.init)`, i.e. Swift `Character`s, which
/// are grapheme CLUSTERS. For scripts that combine code points (Devanagari
/// matras/viramas, Kannada, Sanskrit, emoji ZWJ sequences, ...) a cluster is
/// not a vocabulary entry, so it falls through to per-byte `<0xHH>` fallback
/// and the model sees ~2-3x too many, completely different tokens. English,
/// French and code are unaffected (their clusters are single scalars), which
/// is why this went unnoticed. Measured on EmbeddingGemma 2: Hindi / Kannada /
/// Sanskrit embeddings had cosine 0.73-0.92 to the reference with the library
/// tokenizer and 1.0 with this one. See docs/EMBEDDINGGEMMA2.md.
///
/// Scope: the tokenizer shape used by Gemma-family `tokenizer.json`s - a
/// `Replace " " -> "▁"` normalizer, `model.type == "BPE"` with `byte_fallback`,
/// literal special tokens, and a `TemplateProcessing` single-sequence wrap
/// (`<bos> A <eos>`). Anything else is rejected at load rather than silently
/// mis-tokenised. Vocabulary lookups are keyed by raw UTF-8 bytes, never by
/// `String`, so Swift's canonical-equivalence string equality cannot merge two
/// distinct vocab entries.
public final class CodePointBPETokenizer: @unchecked Sendable {
    public enum LoadError: Error, CustomStringConvertible {
        case unsupported(String)
        public var description: String {
            if case .unsupported(let m) = self { return "CodePointBPETokenizer: \(m)" }
            return ""
        }
    }

    private let vocab: [[UInt8]: Int32]
    private let scalarToId: [UInt32: Int32]
    private let byteToken: [Int32]                 // 256 entries, -1 when absent
    private let merges: [UInt64: (rank: Int32, id: Int32)]
    private let specials: [(bytes: [UInt8], id: Int32)]  // longest first
    public let bosId: Int?
    public let eosId: Int?
    public let vocabSize: Int

    public init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? NSDictionary,
              let model = root["model"] as? NSDictionary,
              (model["type"] as? String) == "BPE",
              let vocabDict = model["vocab"] as? NSDictionary,
              let mergeList = model["merges"] as? NSArray else {
            throw LoadError.unsupported("tokenizer.json is not a BPE tokenizer")
        }
        guard (model["byte_fallback"] as? Bool) == true else {
            throw LoadError.unsupported("only byte_fallback BPE is supported")
        }
        if let n = root["normalizer"] as? NSDictionary {
            let pat = (n["pattern"] as? NSDictionary)?["String"] as? String
            guard (n["type"] as? String) == "Replace", pat == " ",
                  (n["content"] as? String) == "\u{2581}" else {
                throw LoadError.unsupported("unsupported normalizer (expected Replace ' ' -> '\u{2581}')")
            }
        } else {
            throw LoadError.unsupported("expected a Replace ' ' -> '\u{2581}' normalizer")
        }

        // vocab: token (as raw utf8) -> id
        var idBytes = [[UInt8]](repeating: [], count: vocabDict.count)
        var vocab = [[UInt8]: Int32](minimumCapacity: vocabDict.count)
        var scalarToId: [UInt32: Int32] = [:]
        var maxId = -1
        for (k, v) in vocabDict {
            guard let ks = k as? NSString, let id = (v as? NSNumber)?.intValue else { continue }
            let bytes = Array((ks as String).utf8)
            vocab[bytes] = Int32(id)
            if id >= idBytes.count { idBytes += [[UInt8]](repeating: [], count: id - idBytes.count + 1) }
            idBytes[id] = bytes
            maxId = max(maxId, id)
            var it = (ks as String).unicodeScalars.makeIterator()
            if let first = it.next(), it.next() == nil { scalarToId[first.value] = Int32(id) }
        }
        var byteToken = [Int32](repeating: -1, count: 256)
        for b in 0 ..< 256 {
            let key = Array(String(format: "<0x%02X>", b).utf8)
            if let id = vocab[key] { byteToken[b] = id }
        }

        // merges: (idA, idB) -> (rank, id of a+b)
        var merges = [UInt64: (rank: Int32, id: Int32)](minimumCapacity: mergeList.count)
        for (rank, m) in mergeList.enumerated() {
            guard let pair = m as? NSArray, pair.count == 2,
                  let a = pair[0] as? String, let b = pair[1] as? String,
                  let ia = vocab[Array(a.utf8)], let ib = vocab[Array(b.utf8)],
                  let merged = vocab[Array(a.utf8) + Array(b.utf8)] else { continue }
            let key = (UInt64(UInt32(bitPattern: ia)) << 32) | UInt64(UInt32(bitPattern: ib))
            if merges[key] == nil { merges[key] = (Int32(rank), merged) }
        }

        // literal special tokens ("added_tokens" with special == true)
        var specials: [(bytes: [UInt8], id: Int32)] = []
        for t in (root["added_tokens"] as? NSArray) ?? [] {
            guard let d = t as? NSDictionary, (d["special"] as? Bool) == true,
                  let c = d["content"] as? String, let id = (d["id"] as? NSNumber)?.int32Value,
                  !c.isEmpty else { continue }
            specials.append((Array(c.utf8), id))
        }
        specials.sort { $0.bytes.count > $1.bytes.count }

        // TemplateProcessing "single": <bos> A <eos>
        var bos: Int? = nil, eos: Int? = nil
        if let pp = root["post_processor"] as? NSDictionary,
           (pp["type"] as? String) == "TemplateProcessing",
           let single = pp["single"] as? NSArray {
            var sawSeq = false
            for e in single {
                guard let d = e as? NSDictionary else { continue }
                if d["Sequence"] != nil { sawSeq = true; continue }
                guard let st = d["SpecialToken"] as? NSDictionary, let tok = st["id"] as? String,
                      let id = vocab[Array(tok.utf8)] ?? specials.first(where: { $0.bytes == Array(tok.utf8) })?.id
                else { continue }
                if sawSeq { eos = Int(id) } else { bos = Int(id) }
            }
        }

        self.vocab = vocab
        self.scalarToId = scalarToId
        self.byteToken = byteToken
        self.merges = merges
        self.specials = specials
        self.bosId = bos
        self.eosId = eos
        self.vocabSize = vocabDict.count
    }

    /// Encode `text`. With `addSpecialTokens` the template's `<bos>`/`<eos>`
    /// wrap is applied (matches `tokenizer(text)["input_ids"]` in HF).
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        var out: [Int] = []
        if addSpecialTokens, let b = bosId { out.append(b) }

        // Split on literal special tokens (they are matched in the raw text,
        // before normalisation), BPE-encode the rest.
        let bytes = Array(text.utf8)
        var segStart = 0
        var i = 0
        func flush(_ end: Int) {
            if end > segStart {
                let seg = String(decoding: bytes[segStart ..< end], as: UTF8.self)
                out.append(contentsOf: encodeSegment(seg))
            }
        }
        while i < bytes.count {
            if bytes[i] == 0x3C /* < */, let hit = matchSpecial(bytes, at: i) {
                flush(i)
                out.append(Int(hit.id))
                i += hit.bytes.count
                segStart = i
            } else {
                i += 1
            }
        }
        flush(bytes.count)

        if addSpecialTokens, let e = eosId { out.append(e) }
        return out
    }

    private func matchSpecial(_ bytes: [UInt8], at i: Int) -> (bytes: [UInt8], id: Int32)? {
        for s in specials where i + s.bytes.count <= bytes.count && bytes[i] == s.bytes[0] {
            if bytes[i ..< i + s.bytes.count].elementsEqual(s.bytes) { return s }
        }
        return nil
    }

    private func encodeSegment(_ text: String) -> [Int] {
        // normalizer: " " -> "▁"; then one symbol per Unicode scalar.
        var syms: [Int32] = []
        syms.reserveCapacity(text.unicodeScalars.count)
        for sc in text.unicodeScalars {
            let v = sc.value == 0x20 ? 0x2581 : sc.value
            if let id = scalarToId[v] {
                syms.append(id)
            } else {
                // byte fallback: one <0xHH> symbol per UTF-8 byte
                for b in String(sc).utf8 {
                    let id = byteToken[Int(b)]
                    if id >= 0 { syms.append(id) }
                }
            }
        }
        return mergeAll(syms).map { Int($0) }
    }

    // MARK: - BPE merge (rank-ordered, leftmost first)

    private struct Cand { var rank: Int32; var pos: Int32; var l: Int32; var r: Int32 }

    private func mergeAll(_ ids: [Int32]) -> [Int32] {
        let n = ids.count
        if n < 2 { return ids }
        var sym = ids
        var prev = [Int32](repeating: -1, count: n)
        var next = [Int32](repeating: -1, count: n)
        var alive = [Bool](repeating: true, count: n)
        for i in 0 ..< n {
            prev[i] = Int32(i - 1)
            next[i] = i + 1 < n ? Int32(i + 1) : -1
        }
        var heap: [Cand] = []
        heap.reserveCapacity(n)

        @inline(__always) func less(_ a: Cand, _ b: Cand) -> Bool {
            a.rank != b.rank ? a.rank < b.rank : a.pos < b.pos
        }
        func push(_ c: Cand) {
            heap.append(c)
            var k = heap.count - 1
            while k > 0 {
                let p = (k - 1) / 2
                if less(heap[k], heap[p]) { heap.swapAt(k, p); k = p } else { break }
            }
        }
        func pop() -> Cand? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var k = 0
                while true {
                    let l = 2 * k + 1, r = l + 1
                    var m = k
                    if l < heap.count, less(heap[l], heap[m]) { m = l }
                    if r < heap.count, less(heap[r], heap[m]) { m = r }
                    if m == k { break }
                    heap.swapAt(k, m); k = m
                }
            }
            return top
        }
        @inline(__always) func key(_ a: Int32, _ b: Int32) -> UInt64 {
            (UInt64(UInt32(bitPattern: a)) << 32) | UInt64(UInt32(bitPattern: b))
        }
        func tryPush(_ i: Int) {
            let j = Int(next[i])
            guard j >= 0, let m = merges[key(sym[i], sym[j])] else { return }
            push(Cand(rank: m.rank, pos: Int32(i), l: sym[i], r: sym[j]))
        }
        for i in 0 ..< n - 1 { tryPush(i) }

        while let c = pop() {
            let i = Int(c.pos)
            guard alive[i], sym[i] == c.l else { continue }
            let j = Int(next[i])
            guard j >= 0, alive[j], sym[j] == c.r, let m = merges[key(c.l, c.r)] else { continue }
            sym[i] = m.id
            alive[j] = false
            let nn = next[j]
            next[i] = nn
            if nn >= 0 { prev[Int(nn)] = Int32(i) }
            if prev[i] >= 0 { tryPush(Int(prev[i])) }
            tryPush(i)
        }

        var out: [Int32] = []
        out.reserveCapacity(n)
        var k = 0
        while k >= 0 && k < n { out.append(sym[k]); k = Int(next[k]) }
        return out
    }
}
