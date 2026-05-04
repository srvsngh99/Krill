import Foundation

/// Python mlx-vlm fallback for complex models (Gemma 4) where native Swift
/// implementation is still being debugged.
///
/// Shells out to `python3 -c "from mlx_vlm import ..."` for inference.
/// This is a temporary bridge until the native Swift implementation is perfected.
public final class PythonFallback: @unchecked Sendable {

    private let modelPath: String

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    /// Python executable path (checks venv and system).
    private static let pythonPath: String? = {
        // Check known venv first
        let venvPython = "/private/tmp/mlx_debug_env/bin/python3"
        if FileManager.default.fileExists(atPath: venvPython) {
            return venvPython
        }
        // Check system python
        return "/usr/bin/env"
    }()

    private static let pythonArgs: [String] = {
        let venvPython = "/private/tmp/mlx_debug_env/bin/python3"
        if FileManager.default.fileExists(atPath: venvPython) {
            return []  // venvPython is the full path
        }
        return ["python3"]  // env will find python3
    }()

    /// Check if Python mlx-vlm is available.
    public static var isAvailable: Bool {
        guard let python = pythonPath else { return false }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = pythonArgs + ["-c", "from mlx_vlm import load; print('ok')"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
        return process.terminationStatus == 0
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
        let escapedPrompt = prompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")

        var generateArgs = ""
        if let img = imagePath {
            generateArgs += ", image=\"\(img)\""
        }
        if let audio = audioPath {
            generateArgs += ", audio=\"\(audio)\""
        }

        // Build multimodal prefix tokens
        var mediaPrefix = ""
        if imagePath != nil { mediaPrefix += "<|image|>" }
        if audioPath != nil { mediaPrefix += "<|audio|>" }

        var kwargs: [String] = []
        if let img = imagePath { kwargs.append("image=[\"\(img)\"]") }
        if let audio = audioPath { kwargs.append("audio=\"\(audio)\"") }
        let kwargsStr = kwargs.isEmpty ? "" : ", " + kwargs.joined(separator: ", ")

        let script = """
        from mlx_vlm import load, generate
        model, processor = load("\(modelPath)")
        tok = processor.tokenizer
        bos = tok.decode([2])
        turn_start = tok.decode([105])
        turn_end = tok.decode([106])
        newline = tok.decode([107])
        prompt = f'{bos}{turn_start}user{newline}\(mediaPrefix)\(escapedPrompt){turn_end}{newline}{turn_start}model{newline}'
        result = generate(model, processor, prompt=prompt, max_tokens=\(maxTokens)\(kwargsStr), verbose=False)
        text = result.text if hasattr(result, 'text') else str(result)
        print(text.strip())
        """

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
