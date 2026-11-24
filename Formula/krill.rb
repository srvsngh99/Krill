# Release snapshot for this repository's consistency checks. The installable
# formula (with its bottle) lives in github.com/srvsngh99/homebrew-krill.
# Install: brew tap srvsngh99/krill && brew install krill
class Krill < Formula
  desc "Local LLM engine and coding agent in one Mac-native binary"
  homepage "https://github.com/srvsngh99/Krill"
  url "https://github.com/srvsngh99/Krill/releases/download/v0.28.1/krill-0.28.1-arm64-apple-macos.tar.gz"
  sha256 "d97244ca5ec0d4d6b1612bd470bcd14f5ea869eb6b319bbb51545a01608b4a3a"
  license "MIT"
  version "0.28.1"

  depends_on :macos
  depends_on arch: :arm64

  def install
    # MLX looks for mlx.metallib next to the executable first, so the
    # metallib and Cmlx bundle sit beside the binary in libexec/.
    libexec.install "krill"
    libexec.install "mlx.metallib" if File.exist?("mlx.metallib")
    if Dir.exist?("mlx-swift_Cmlx.bundle")
      libexec.install "mlx-swift_Cmlx.bundle"
    end
    bin.install_symlink libexec/"krill"
  end

  def caveats
    <<~EOS
      >_ Krill - the model and the agent, in one Mac-native binary.  (Apple Silicon, M1+)

      Run a model    krill pull gemma-4-e2b && krill run gemma-4-e2b
      Give it work   krill code "fix the failing test"     # agent: bash, edits, web
      Serve it       krill serve                           # OpenAI/Ollama/Anthropic API + /ui on your phone
      Lend it        krill launch claude                   # point Claude Code/Codex at it

      Config: ~/.krill/config.toml    Models: ~/.krill/models/

      a Sourav AI Labs project - souravailabs.ai
    EOS
  end

  test do
    assert_match "Krill", shell_output("#{bin}/krill version")
  end
end
