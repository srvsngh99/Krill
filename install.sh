#!/bin/sh
# Krill one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh
#
# Downloads the latest signed Krill release, lays the `krill` binary down next
# to its `mlx.metallib` and Metal bundle (the MLX metallib loader looks in the
# executable's directory first, so they must sit together — same layout as the
# Homebrew formula), and links `krill` onto your PATH. Apple Silicon only.
#
# Environment overrides:
#   KRILL_PREFIX   install prefix (default: /usr/local). Binary lands in
#                  $KRILL_PREFIX/bin, resources in $KRILL_PREFIX/libexec/krill.
#   KRILL_VERSION  a specific version to install (e.g. 0.12.0); default: latest.

set -eu

REPO="srvsngh99/Krill"
PREFIX="${KRILL_PREFIX:-/usr/local}"
LIBDIR="$PREFIX/libexec/krill"
BINDIR="$PREFIX/bin"

err() { printf 'krill-install: %s\n' "$1" >&2; exit 1; }
info() { printf '%s\n' "$1"; }

# 1. Platform + tooling checks ------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || err "Krill requires macOS (Apple Silicon)."
[ "$(uname -m)" = "arm64" ] || err "Krill requires Apple Silicon (arm64); detected $(uname -m)."
command -v curl >/dev/null 2>&1 || err "curl is required but not found."
command -v tar >/dev/null 2>&1 || err "tar is required but not found."

# 2. Resolve the release asset URL -------------------------------------------
VERSION="${KRILL_VERSION:-latest}"
if [ "$VERSION" = "latest" ]; then
  API="https://api.github.com/repos/$REPO/releases/latest"
else
  API="https://api.github.com/repos/$REPO/releases/tags/v${VERSION#v}"
fi
info "Resolving Krill release ($VERSION)..."
ASSET_URL=$(curl -fsSL "$API" \
  | grep -o 'https://[^"]*krill-[^"]*-arm64-apple-macos\.tar\.gz' \
  | head -n 1)
[ -n "$ASSET_URL" ] || err "Could not find an arm64 release tarball. Try setting KRILL_VERSION."

# 3. Download + extract to a temp dir ----------------------------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM
info "Downloading $ASSET_URL"
curl -fSL --progress-bar "$ASSET_URL" -o "$TMP/krill.tar.gz" || err "download failed."
tar -xzf "$TMP/krill.tar.gz" -C "$TMP" || err "could not extract the tarball."
[ -f "$TMP/krill" ] || err "tarball did not contain the krill binary."

# 4. Install (escalate with sudo only if the prefix is not writable) ---------
SUDO=""
if [ ! -w "$PREFIX" ] || { [ -d "$BINDIR" ] && [ ! -w "$BINDIR" ]; }; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    info "Installing to $PREFIX (requires sudo)..."
  else
    err "$PREFIX is not writable and sudo is unavailable. Set KRILL_PREFIX to a writable directory."
  fi
fi
$SUDO mkdir -p "$LIBDIR" "$BINDIR"
$SUDO rm -rf "$LIBDIR/krill" "$LIBDIR/mlx.metallib" "$LIBDIR/mlx-swift_Cmlx.bundle"
$SUDO cp "$TMP/krill" "$LIBDIR/krill"
$SUDO chmod +x "$LIBDIR/krill"
[ -f "$TMP/mlx.metallib" ] && $SUDO cp "$TMP/mlx.metallib" "$LIBDIR/mlx.metallib"
[ -d "$TMP/mlx-swift_Cmlx.bundle" ] && $SUDO cp -R "$TMP/mlx-swift_Cmlx.bundle" "$LIBDIR/mlx-swift_Cmlx.bundle"

# 5. Strip Gatekeeper quarantine off the downloaded files --------------------
$SUDO xattr -dr com.apple.quarantine "$LIBDIR" 2>/dev/null || true

# 6. Link onto PATH ----------------------------------------------------------
$SUDO ln -sf "$LIBDIR/krill" "$BINDIR/krill"

info ""
info "Krill installed: $BINDIR/krill"
case ":$PATH:" in
  *":$BINDIR:"*) : ;;
  *) info "Note: $BINDIR is not on your PATH. Add it with:  export PATH=\"$BINDIR:\$PATH\"" ;;
esac
info "Verify with 'krill version', then 'krill pull gemma-4-e2b' to grab a model."
