#!/usr/bin/env python3
"""Offline tests for install.sh release-digest enforcement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER = REPO_ROOT / "install.sh"
ASSET_NAME = "krill-9.9.9-arm64-apple-macos.tar.gz"
ASSET_URL = f"https://github.com/srvsngh99/Krill/releases/download/v9.9.9/{ASSET_NAME}"


class InstallerDigestTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, published_digest: str | None = None) -> tuple[Path, Path]:
        payload = root / "payload"
        payload.mkdir()
        binary = payload / "krill"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

        archive = root / ASSET_NAME
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(binary, arcname="krill")

        actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        metadata = root / "release.json"
        metadata.write_text(
            json.dumps(
                {
                    "tag_name": "v9.9.9",
                    "assets": [
                        {
                            "name": ASSET_NAME,
                            "digest": f"sha256:{published_digest or actual_digest}",
                            "browser_download_url": ASSET_URL,
                        }
                    ]
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return archive, metadata

    def run_installer(self, root: Path, archive: Path, metadata: Path) -> subprocess.CompletedProcess[str]:
        mock_bin = root / "bin"
        mock_bin.mkdir()

        uname = mock_bin / "uname"
        uname.write_text(
            '#!/bin/sh\n[ "$1" = "-s" ] && printf "Darwin\\n" || printf "arm64\\n"\n',
            encoding="utf-8",
        )
        uname.chmod(0o755)

        curl = mock_bin / "curl"
        curl.write_text(
            "#!/bin/sh\n"
            "out=\nurl=\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  case \"$1\" in\n"
            "    -o) shift; out=$1 ;;\n"
            "    -*) ;;\n"
            "    *) url=$1 ;;\n"
            "  esac\n"
            "  shift\n"
            "done\n"
            f"case \"$url\" in *api.github.com*) cp \"{metadata}\" \"$out\" ;; "
            f"*) cp \"{archive}\" \"$out\" ;; esac\n",
            encoding="utf-8",
        )
        curl.chmod(0o755)

        prefix = root / "prefix"
        prefix.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{mock_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
                "KRILL_PREFIX": str(prefix),
                "KRILL_VERSION": "9.9.9",
            }
        )
        return subprocess.run(
            ["/bin/sh", str(INSTALLER)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_verified_archive_installs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krill-installer-test-") as temp:
            root = Path(temp)
            archive, metadata = self.make_fixture(root)
            result = self.run_installer(root, archive, metadata)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Verified SHA-256", result.stdout)
            self.assertTrue((root / "prefix/libexec/krill/krill").is_file())

    def test_digest_mismatch_fails_before_extraction_or_install(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krill-installer-test-") as temp:
            root = Path(temp)
            archive, metadata = self.make_fixture(root, published_digest="0" * 64)
            result = self.run_installer(root, archive, metadata)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 verification failed", result.stderr)
            self.assertFalse((root / "prefix/libexec/krill/krill").exists())

    def test_missing_digest_does_not_borrow_digest_from_another_asset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krill-installer-test-") as temp:
            root = Path(temp)
            archive, metadata = self.make_fixture(root)
            metadata.write_text(
                json.dumps(
                    {
                        "tag_name": "v9.9.9",
                        "assets": [
                            {"name": ASSET_NAME, "browser_download_url": ASSET_URL},
                            {
                                "name": "unrelated.tar.gz",
                                "digest": "sha256:" + "0" * 64,
                                "browser_download_url": "https://example.invalid/unrelated.tar.gz",
                            },
                        ]
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            result = self.run_installer(root, archive, metadata)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no valid published SHA-256 digest", result.stderr)
            self.assertFalse((root / "prefix/libexec/krill/krill").exists())

    def test_selects_exact_asset_for_release_tag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krill-installer-test-") as temp:
            root = Path(temp)
            archive, metadata = self.make_fixture(root)
            actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            metadata.write_text(
                json.dumps(
                    {
                        "tag_name": "v9.9.9",
                        "assets": [
                            {
                                "name": "krill-8.8.8-arm64-apple-macos.tar.gz",
                                "digest": "sha256:" + "0" * 64,
                                "browser_download_url": "https://example.invalid/wrong.tar.gz",
                            },
                            {
                                "name": ASSET_NAME,
                                "digest": "sha256:" + actual_digest,
                                "browser_download_url": ASSET_URL,
                            },
                        ],
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            result = self.run_installer(root, archive, metadata)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "prefix/libexec/krill/krill").is_file())


if __name__ == "__main__":
    unittest.main()
