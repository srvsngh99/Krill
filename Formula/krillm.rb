# Homebrew formula for KrillLM
# Install: brew tap srvsngh99/krillm && brew install krillm
class Krillm < Formula
  desc "Fast local LLM inference CLI for Apple Silicon"
  homepage "https://github.com/srvsngh99/KrillLM"
  url "https://github.com/srvsngh99/KrillLM/releases/download/v0.2.0/krillm-0.2.0-arm64-apple-macos.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"
  version "0.2.0"

  depends_on :macos
  depends_on arch: :arm64

  def install
    bin.install "krillm"
  end

  def caveats
    <<~EOS
      KrillLM requires Apple Silicon (M1 or newer).

      Quick start:
        krillm pull llama-3.2-3b
        krillm run llama-3.2-3b

      Start the API server (Ollama/OpenAI compatible):
        krillm serve --model llama-3.2-3b

      Configuration: ~/.krillm/config.toml
      Models stored: ~/.krillm/models/
    EOS
  end

  test do
    assert_match "KrillLM", shell_output("#{bin}/krillm version")
  end
end
