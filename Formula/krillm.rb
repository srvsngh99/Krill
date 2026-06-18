# Homebrew formula for KrillLM
# Install: brew tap srvsngh99/krillm && brew install krillm
class Krillm < Formula
  desc "Fast local LLM inference CLI for Apple Silicon"
  homepage "https://github.com/srvsngh99/KrillLM"
  url "https://github.com/srvsngh99/KrillLM/releases/download/v0.10.0/krillm-0.10.0-arm64-apple-macos.tar.gz"
  sha256 "c10bc087797b473856b75346fbdcb6205eb6ded3133ad044a378989a0cfff896"
  license "MIT"
  version "0.10.0"

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
    libexec.install "krillm"
    libexec.install "mlx.metallib" if File.exist?("mlx.metallib")
    if Dir.exist?("mlx-swift_Cmlx.bundle")
      libexec.install "mlx-swift_Cmlx.bundle"
    end
    bin.install_symlink libexec/"krillm"
  end

  def caveats
    <<~EOS
      >_ KrillLM - a fast, lean LLM runtime, built for Mac.  (Apple Silicon, M1+)

      Get started:
        krillm pull gemma-4-e2b      # a small, fast model to begin
        krillm run gemma-4-e2b       # open the chat

      Serve an Ollama/OpenAI-compatible API:
        krillm serve --model gemma-4-e2b

      Config: ~/.krillm/config.toml    Models: ~/.krillm/models/

      a Sourav AI Labs project - souravailabs.ai
    EOS
  end

  test do
    assert_match "KrillLM", shell_output("#{bin}/krillm version")
  end
end
