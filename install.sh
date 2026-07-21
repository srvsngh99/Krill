#!/bin/sh
# KRILL_INSTALLER_VERIFIES_SHA256=1
# Krill one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh
#
# Downloads the latest published Krill release, verifies its GitHub-published
# SHA-256 digest, lays the `krill` binary down next
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
command -v shasum >/dev/null 2>&1 || err "shasum is required but not found."
command -v plutil >/dev/null 2>&1 || err "plutil is required but not found."

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

# 2. Resolve the release asset URL + published digest ------------------------
VERSION="${KRILL_VERSION:-latest}"
if [ "$VERSION" = "latest" ]; then
  API="https://api.github.com/repos/$REPO/releases/latest"
else
  API="https://api.github.com/repos/$REPO/releases/tags/v${VERSION#v}"
fi
info "Resolving Krill release ($VERSION)..."
curl -fsSL "$API" -o "$TMP/release.json" \
  || err "Could not fetch release metadata from GitHub."
RELEASE_TAG=$(plutil -extract tag_name raw -o - "$TMP/release.json" 2>/dev/null || true)
case "$RELEASE_TAG" in
  v*) RESOLVED_VERSION=${RELEASE_TAG#v} ;;
  *) err "Release metadata did not contain a valid v-prefixed tag." ;;
esac
case "$RESOLVED_VERSION" in
  ''|*[!0-9.]*|.*|*..*|*.) err "Release metadata contained an invalid version tag." ;;
esac
OLD_IFS=$IFS
IFS=.
set -- $RESOLVED_VERSION
IFS=$OLD_IFS
[ "$#" -eq 3 ] || err "Release metadata contained an invalid version tag."
if [ "$VERSION" != "latest" ] && [ "$RELEASE_TAG" != "v${VERSION#v}" ]; then
  err "Release tag did not match requested version ${VERSION#v}."
fi
EXPECTED_ASSET_NAME="krill-$RESOLVED_VERSION-arm64-apple-macos.tar.gz"
ASSET_INDEX=0
ASSET_NAME=""
ASSET_URL=""
ASSET_DIGEST=""
while NAME=$(plutil -extract "assets.$ASSET_INDEX.name" raw -o - "$TMP/release.json" 2>/dev/null); do
  if [ "$NAME" = "$EXPECTED_ASSET_NAME" ]; then
    ASSET_NAME=$NAME
    ASSET_URL=$(plutil -extract "assets.$ASSET_INDEX.browser_download_url" raw -o - "$TMP/release.json" 2>/dev/null || true)
    ASSET_DIGEST=$(plutil -extract "assets.$ASSET_INDEX.digest" raw -o - "$TMP/release.json" 2>/dev/null || true)
    break
  fi
  ASSET_INDEX=$((ASSET_INDEX + 1))
done
[ -n "$ASSET_URL" ] \
  || err "Could not find the exact release asset $EXPECTED_ASSET_NAME."

# GitHub's release API publishes a `digest` on each uploaded asset. Because
# plutil addresses the selected asset by array index, a missing digest can
# never be accidentally borrowed from a neighboring asset.
case "$ASSET_DIGEST" in
  sha256:*) EXPECTED_SHA256=${ASSET_DIGEST#sha256:} ;;
  *) EXPECTED_SHA256="" ;;
esac
case "$EXPECTED_SHA256" in
  ''|*[!0-9a-fA-F]*) err "Release asset has no valid published SHA-256 digest; refusing to install." ;;
esac
[ "${#EXPECTED_SHA256}" -eq 64 ] \
  || err "Release asset has an invalid SHA-256 digest; refusing to install."

# 3. Download, verify, and extract -------------------------------------------
info "Downloading $ASSET_URL"
curl -fSL --progress-bar "$ASSET_URL" -o "$TMP/krill.tar.gz" || err "download failed."
ACTUAL_SHA256=$(shasum -a 256 "$TMP/krill.tar.gz" | awk '{print $1}')
EXPECTED_SHA256=$(printf '%s' "$EXPECTED_SHA256" | tr '[:upper:]' '[:lower:]')
ACTUAL_SHA256=$(printf '%s' "$ACTUAL_SHA256" | tr '[:upper:]' '[:lower:]')
[ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ] \
  || err "SHA-256 verification failed for $ASSET_NAME (expected $EXPECTED_SHA256, got $ACTUAL_SHA256)."
info "Verified SHA-256: $EXPECTED_SHA256"
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
