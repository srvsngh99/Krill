# Homebrew formula for KrillLM
# Install: brew tap srvsngh99/krillm && brew install krillm
class Krillm < Formula
  desc "Fast local LLM inference CLI for Apple Silicon"
  homepage "https://github.com/srvsngh99/KrillLM"
  url "https://github.com/srvsngh99/KrillLM/releases/download/v0.5.1/krillm-0.5.1-arm64-apple-macos.tar.gz"
  sha256 "099921d2005aa374ff69ea3fe787bf0b8c3bbde68293e8cf9e54ed0a73340c30"
  license "MIT"
  version "0.5.1"

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
