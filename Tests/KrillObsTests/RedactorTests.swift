import XCTest
@testable import KrillObs

final class RedactorTests: XCTestCase {
    let r = Redactor(home: "/Users/alice")

    // Secrets planted in the shapes Krill, users and tools actually produce.
    let planted: [String] = [
        "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "sk-proj-1234567890abcdefghijklmnop",
        "hf_abcdefghijklmnopqrstuvwxyz012345",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
    ]

    func testBearerTokensAndAuthorizationHeaders() {
        let s = r.redact("Authorization: Bearer krill-live-key-9f8e7d6c5b4a39281706 and header Bearer abcdef123456")
        XCTAssertFalse(s.contains("krill-live-key"))
        XCTAssertFalse(s.contains("abcdef123456"))
        XCTAssertTrue(s.contains("Bearer [REDACTED]"))
    }

    func testEveryPlantedTokenShapeIsRemovedFromFreeText() {
        for secret in planted {
            let out = r.redact("log line: failed with credential \(secret) while loading")
            XCTAssertFalse(out.contains(secret), "leaked: \(secret) -> \(out)")
        }
    }

    func testServerAPIKeyAndSecretNamedAssignmentsInConfig() {
        let toml = """
        default_model = "gemma"
        server_api_key = "super-secret-token-123456"
        brave_api_key = "BSA-abcdef"
        tavily_api_key = ""
        idle_timeout = "300"
        models_dir = "/Users/alice/models"
        """
        let out = r.redactConfig(toml)
        XCTAssertFalse(out.contains("super-secret-token-123456"))
        XCTAssertFalse(out.contains("BSA-abcdef"))
        XCTAssertTrue(out.contains("server_api_key = \"[REDACTED]\""))
        XCTAssertTrue(out.contains("tavily_api_key = \"\""), "an unset key stays visibly unset")
        XCTAssertTrue(out.contains("default_model = \"gemma\""))
        XCTAssertTrue(out.contains("idle_timeout = \"300\""))
        XCTAssertTrue(out.contains("models_dir = \"~/models\""))
    }

    func testJSONAssignmentsAreRedactedButTokenCountsAreNot() {
        let line = #"{"event":"request","api_key":"abc123XYZ","prompt_tokens":42,"completion_tokens":7,"max_tokens":128,"session_id":"s-1"}"#
        let out = r.redact(line)
        XCTAssertFalse(out.contains("abc123XYZ"))
        XCTAssertTrue(out.contains("\"prompt_tokens\":42"))
        XCTAssertTrue(out.contains("\"completion_tokens\":7"))
        XCTAssertTrue(out.contains("\"max_tokens\":128"))
        XCTAssertTrue(out.contains("\"session_id\":\"s-1\""))
    }

    func testEnvValuesThatLookSecretAreHiddenByNameOrByShape() {
        let env = [
            "KRILL_API_KEY": "plain",
            "KRILL_BRAVE_API_KEY": "BSA123",
            "KRILL_LOG_LEVEL": "debug",
            "KRILL_SOMETHING": "Zx9Qm2Lp7Vb4Nc1Tr8Ks5Wd3Hf6Jg0A",   // not secret-named, but looks like a token
            "KRILL_MODELS_DIR": "/Users/alice/models",
            "OLLAMA_HOST": "127.0.0.1:11434",
            "PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "nope",
        ]
        let out = Dictionary(uniqueKeysWithValues: r.redactedEnvironment(env))
        XCTAssertEqual(out["KRILL_API_KEY"], "[REDACTED]")
        XCTAssertEqual(out["KRILL_BRAVE_API_KEY"], "[REDACTED]")
        XCTAssertEqual(out["KRILL_SOMETHING"], "[REDACTED]")
        XCTAssertEqual(out["KRILL_LOG_LEVEL"], "debug")
        XCTAssertEqual(out["OLLAMA_HOST"], "127.0.0.1:11434")
        XCTAssertEqual(out["KRILL_MODELS_DIR"], "~/models")
        XCTAssertNil(out["PATH"], "only KRILL_/OLLAMA_/HF_ variables are even considered")
        XCTAssertNil(out["AWS_SECRET_ACCESS_KEY"])
    }

    func testHomePathBecomesTildeEvenForOtherUsers() {
        let out = r.redact("loaded /Users/alice/.krill/models/x and /Users/bob/work and /home/carol/p; cwd=/Users/alice")
        XCTAssertEqual(out, "loaded ~/.krill/models/x and ~/work and ~/p; cwd=~")
    }

    func testEmailAndPrivateKeyBlocks() {
        let out = r.redact("from someone@example.com\n-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\nBBBB\n-----END OPENSSH PRIVATE KEY-----\nend")
        XCTAssertFalse(out.contains("someone@example.com"))
        XCTAssertFalse(out.contains("AAAA"))
        XCTAssertTrue(out.hasSuffix("end"))
    }

    func testOrdinaryTextIsLeftAlone() {
        let text = "model gemma-4-e2b loaded in 1532 ms, 4096 tokens, GET /v1/chat/completions 200"
        XCTAssertEqual(r.redact(text), text)
    }
}
