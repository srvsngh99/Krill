import MLX

/// MLX/Metal memory counters. The process's resident size does not include Metal buffers that
/// hold model weights, so `/metrics` and `/v1/status` report these next to the footprint.
public enum EngineMemory {
    /// Bytes currently held by live MLX arrays (weights, KV cache, activations).
    public static var activeBytes: Int { Memory.activeMemory }
    /// High-water mark of `activeBytes` since process start (or the last reset).
    public static var peakBytes: Int { Memory.peakMemory }
    /// Bytes MLX keeps cached for reuse.
    public static var cacheBytes: Int { Memory.cacheMemory }
}
