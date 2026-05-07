import Foundation

/// Python mlx-vlm fallback for complex models (Gemma 4) where native Swift
/// implementation is still being debugged.
///
/// Shells out to `python3 -c "from mlx_vlm import ..."` for inference.
/// This is a temporary bridge until the native Swift implementation is perfected.
///
/// Python resolution order:
///   1. `KRILLM_PYTHON` environment variable (explicit override)
///   2. `~/.krillm/venv/bin/python3` (managed venv)
///   3. System `python3` via `/usr/bin/env`
public final class PythonFallback: @unchecked Sendable {

    private let modelPath: String

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    public struct Availability: Equatable, Sendable {
        public let pythonCommand: String
        public let isAvailable: Bool
        public let detail: String
    }

    /// Candidate venv paths to search, in priority order.
    private static let venvSearchPaths: [String] = {
        var paths: [String] = []
        // 1. Explicit override from environment
        if let envPython = ProcessInfo.processInfo.environment["KRILLM_PYTHON"] {
            paths.append(envPython)
        }
        // 2. KrillLM managed venv
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        paths.append("\(home)/.krillm/venv/bin/python3")
        // 3. Common mlx venv locations
        paths.append("\(home)/.venv/bin/python3")
        paths.append("/opt/homebrew/bin/python3")
        return paths
    }()

    /// Resolved Python executable path. Checks venvs first, falls back to system python3.
    private static let pythonPath: String? = {
        for path in venvSearchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        // Fallback: system python3 via env
        return "/usr/bin/env"
    }()

    private static let pythonArgs: [String] = {
        for path in venvSearchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return []  // direct path, no args needed
            }
        }
        return ["python3"]  // env will find python3
    }()

    /// Check if Python mlx-vlm is available.
    public static var isAvailable: Bool {
        checkAvailability().isAvailable
    }

    /// Check Python and mlx-vlm availability, including a human-readable reason.
    public static func checkAvailability() -> Availability {
        guard let python = pythonPath else {
            return Availability(
                pythonCommand: "",
                isAvailable: false,
                detail: "No Python executable found")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = pythonArgs + ["-c", "from mlx_vlm import load; print('ok')"]
        process.standardOutput = FileHandle.nullDevice
        let errPipe = Pipe()
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return Availability(
                pythonCommand: ([python] + pythonArgs).joined(separator: " "),
                isAvailable: false,
                detail: "Unable to run Python: \(error.localizedDescription)")
        }
        process.waitUntilExit()
        if process.terminationStatus == 0 {
            return Availability(
                pythonCommand: ([python] + pythonArgs).joined(separator: " "),
                isAvailable: true,
                detail: "mlx-vlm available")
        }
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let err = String(data: errData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = err?.isEmpty == false
            ? err!
            : "Python mlx-vlm not installed (pip install mlx-vlm)"
        return Availability(
            pythonCommand: ([python] + pythonArgs).joined(separator: " "),
            isAvailable: false,
            detail: detail)
    }

    /// Generate text using Python mlx-vlm.
    ///
    /// - Parameters:
    ///   - prompt: User prompt
    ///   - systemPrompt: Optional system prompt
    ///   - maxTokens: Maximum tokens to generate
    ///   - imagePath: Optional image path for vision
    ///   - audioPath: Optional audio path for audio understanding
    /// - Returns: Generated text
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        imagePath: String? = nil,
        audioPath: String? = nil
    ) async throws -> String {
        let script = Self.buildScript(
            modelPath: modelPath,
            prompt: prompt,
            maxTokens: maxTokens,
            imagePath: imagePath,
            audioPath: audioPath)

        guard let python = PythonFallback.pythonPath else {
            throw FallbackError.notAvailable
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = PythonFallback.pythonArgs + ["-c", script]

        let pipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = pipe
        process.standardError = errPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
            let err = String(data: errData, encoding: .utf8) ?? "Unknown error"
            throw FallbackError.pythonFailed(err)
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return (String(data: data, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func buildScript(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        imagePath: String?,
        audioPath: String?
    ) -> String {
        var mediaPrefix = ""
        if imagePath != nil { mediaPrefix += "<|image|>" }
        if audioPath != nil { mediaPrefix += "<|audio|>" }

        var kwargsLines: [String] = []
        if let imagePath {
            kwargsLines.append("kwargs[\"image\"] = [\(pythonStringLiteral(imagePath))]")
        }
        if let audioPath {
            kwargsLines.append("kwargs[\"audio\"] = \(pythonStringLiteral(audioPath))")
        }
        let kwargsBlock = kwargsLines.isEmpty
            ? "pass"
            : kwargsLines.joined(separator: "\n")

        return """
        from mlx_vlm import load, generate
        model, processor = load(\(pythonStringLiteral(modelPath)))
        tok = processor.tokenizer
        bos = tok.decode([2])
        turn_start = tok.decode([105])
        turn_end = tok.decode([106])
        newline = tok.decode([107])
        user_prompt = \(pythonStringLiteral(prompt))
        media_prefix = \(pythonStringLiteral(mediaPrefix))
        kwargs = {}
        \(kwargsBlock)
        prompt = f'{bos}{turn_start}user{newline}{media_prefix}{user_prompt}{turn_end}{newline}{turn_start}model{newline}'
        result = generate(model, processor, prompt=prompt, max_tokens=\(maxTokens), verbose=False, **kwargs)
        text = result.text if hasattr(result, 'text') else str(result)
        print(text.strip())
        """
    }

    private static func pythonStringLiteral(_ value: String) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = (try? encoder.encode(value)) ?? Data("\"\"".utf8)
        return String(data: data, encoding: .utf8) ?? "\"\""
    }
}

public enum FallbackError: Error, CustomStringConvertible {
    case pythonFailed(String)
    case notAvailable

    public var description: String {
        switch self {
        case .pythonFailed(let msg): return "Python fallback failed: \(msg.prefix(200))"
        case .notAvailable: return "Python mlx-vlm not installed (pip install mlx-vlm)"
        }
    }
}
