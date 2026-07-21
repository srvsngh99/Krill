import Foundation
import XCTest
@testable import KrillCLI

final class AgentProfilesTests: XCTestCase {
    private let baseURL = "http://127.0.0.1:57455"
    private let model = "test-model"
    private let apiKey = "authenticated-secret"

    func testClaudeAndCodexUseResolvedCredential() {
        XCTAssertEqual(
            AgentProfiles.claude.env(baseURL, model, apiKey)["ANTHROPIC_AUTH_TOKEN"],
            apiKey)
        XCTAssertEqual(
            AgentProfiles.codex.env(baseURL, model, apiKey)["KRILL_API_KEY"],
            apiKey)
    }

    func testOpenCodeReferencesCredentialEnvironment() throws {
        XCTAssertEqual(
            AgentProfiles.opencode.env(baseURL, model, apiKey)["KRILL_API_KEY"],
            apiKey)
        let config = try firstJSONConfig(for: AgentProfiles.opencode)
        let providers = try XCTUnwrap(config["provider"] as? [String: Any])
        let krill = try XCTUnwrap(providers["krill"] as? [String: Any])
        let options = try XCTUnwrap(krill["options"] as? [String: Any])
        XCTAssertEqual(options["apiKey"] as? String, "{env:KRILL_API_KEY}")
    }

    func testHermesPersistsResolvedCredential() {
        let commands = AgentProfiles.hermes.preExec(baseURL, model, apiKey)
        XCTAssertTrue(commands.contains([
            "hermes", "config", "set", "model.api_key", apiKey,
        ]))
    }

    func testPiAndDroidRenderResolvedCredential() throws {
        let piConfig = try firstJSONConfig(for: AgentProfiles.pi)
        let providers = try XCTUnwrap(piConfig["providers"] as? [String: Any])
        let piKrill = try XCTUnwrap(providers["krill"] as? [String: Any])
        XCTAssertEqual(piKrill["apiKey"] as? String, apiKey)

        let droidConfig = try firstJSONConfig(for: AgentProfiles.droid)
        let models = try XCTUnwrap(droidConfig["custom_models"] as? [[String: Any]])
        XCTAssertEqual(models.first?["api_key"] as? String, apiKey)
    }

    func testCopilotUsesDocumentedBYOKCredentialSurface() {
        let env = AgentProfiles.copilot.env(baseURL, model, apiKey)
        XCTAssertEqual(env["COPILOT_PROVIDER_BASE_URL"], baseURL + "/v1")
        XCTAssertEqual(env["COPILOT_PROVIDER_API_KEY"], apiKey)
        XCTAssertEqual(env["COPILOT_PROVIDER_TYPE"], "openai")
        XCTAssertEqual(env["COPILOT_MODEL"], model)
    }

    private func firstJSONConfig(for profile: AgentProfile) throws -> [String: Any] {
        let file = try XCTUnwrap(profile.configFiles.first)
        let rendered = file.render(baseURL, model, apiKey)
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any])
    }
}
