# Release snapshot used by this repository's consistency checks.
# The canonical installable formula lives in github.com/srvsngh99/homebrew-krill.
# Release prep updates both copies from the same published asset and SHA-256.
# Install: brew tap srvsngh99/krill && brew install krill
class Krill < Formula
  desc "Fast local LLM inference CLI for Apple Silicon"
  homepage "https://github.com/srvsngh99/Krill"
  url "https://github.com/srvsngh99/Krill/releases/download/v0.17.0/krill-0.17.0-arm64-apple-macos.tar.gz"
  sha256 "7a8da453678bc5c1984d3db87e4a46b0797ada2c75938add876a41fcd832caec"
  license "MIT"
  version "0.17.0"

  depends_on :macos
  depends_on arch: :arm64

  def install
    # MLX-Swift's metallib loader searches the executable directory
    # first (MLXMetalResourceLocator.candidates[0] is
    # `executableDirectory/mlx.metallib`), so co-locate the metallib
    # and the Cmlx bundle next to the binary in libexec/, then
    # symlink the binary into bin/. Installing the metallib directly
    # into bin/ would also work, but libexec/ keeps Homebrew's
    # `bin` shadowing rules simple and matches the "binary with
    # adjacent resources" convention.
    libexec.install "krill"
    libexec.install "mlx.metallib" if File.exist?("mlx.metallib")
    if Dir.exist?("mlx-swift_Cmlx.bundle")
      libexec.install "mlx-swift_Cmlx.bundle"
    end
    bin.install_symlink libexec/"krill"
  end

  def caveats
    <<~EOS
      >_ Krill - a fast, lean LLM runtime, built for Mac.  (Apple Silicon, M1+)

      Get started:
        krill pull gemma-4-e2b      # a small, fast model to begin
        krill run gemma-4-e2b       # open the chat

      Serve an Ollama/OpenAI-compatible API:
        krill serve --model gemma-4-e2b

      Config: ~/.krill/config.toml    Models: ~/.krill/models/

      a Sourav AI Labs project - souravailabs.ai
    EOS
  end

  test do
    assert_match "Krill", shell_output("#{bin}/krill version")
  end
end
