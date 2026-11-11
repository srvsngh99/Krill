# Release snapshot for this repository's consistency checks. The installable
# formula (with its bottle) lives in github.com/srvsngh99/homebrew-krill.
# Install: brew tap srvsngh99/krill && brew install krill
class Krill < Formula
  desc "Local LLM engine and coding agent in one Mac-native binary"
  homepage "https://github.com/srvsngh99/Krill"
  url "https://github.com/srvsngh99/Krill/releases/download/v0.27.4/krill-0.27.4-arm64-apple-macos.tar.gz"
  sha256 "35c13c550c2d9d34fc4e81bd7c1f3da0026fcc50b942a78a5c5c6277efa56389"
  license "MIT"
  version "0.27.4"

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
