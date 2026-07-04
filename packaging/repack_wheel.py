#!/usr/bin/env python3
"""Repack a vendor MAX wheel with local file overlays + a PEP 440 local tag.

We cannot build a `modular` wheel from source (the compiler, `_core.so`, and
`libmax.so` are closed prebuilt binaries). Instead we take a stock vendor wheel
and bake in the parts we *can* rebuild: our compiled Mojo kernels (`.mojoc`) and
our Python. This keeps the vendor's closed binaries untouched and produces a
genuine `pip install`-able wheel.

The overlay paths are relative to the wheel root, e.g.
  max/nn/quant_ops.py=/abs/path/quant_ops.py                        (max wheel)
  max_mojo_libs-<V>.data/platlib/modular/lib/mojo/linalg.mojoc=/abs/linalg.mojoc

Adds a `+<tag>` local version so pip won't silently replace our wheel with the
stock one on a later resolve; `max==<V>` still matches `<V>+<tag>` (PEP 440), so
`pip install modular==<V>` keeps working. Regenerates `RECORD` exactly.

Usage:
  repack_wheel.py --wheel vendor.whl --out DIR --tag nvfp4sm120.<sha> \
      --overlay <in_wheel_path>=<local_file> [--overlay ...]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


def _hash_and_size(data: bytes) -> tuple[str, int]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
    return "sha256=" + digest.rstrip("="), len(data)


def _parse_wheel_name(name: str) -> tuple[str, str, str]:
    """`max-1.2.3-cp311-cp311-linux.whl` -> (distr, version, rest-tags)."""
    stem = name[:-4] if name.endswith(".whl") else name
    distr, version, rest = stem.split("-", 2)
    return distr, version, rest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="PEP440 local version tag")
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="IN_WHEEL_PATH=LOCAL_FILE",
        help="Overwrite/add a file inside the wheel.",
    )
    args = ap.parse_args()

    distr, version, rest_tags = _parse_wheel_name(args.wheel.name)
    new_version = f"{version}+{args.tag}"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(args.wheel) as zf:
            zf.extractall(root)

        # Apply overlays (relative to wheel root). Version tokens in the target
        # path (e.g. the .data dir) still use the ORIGINAL version.
        for spec in args.overlay:
            in_path, _, local = spec.partition("=")
            dst = root / in_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dst)
            print(f"  overlay: {in_path}")

        # Retag: rename dist-info, bump METADATA Version, then rebuild RECORD.
        old_di = root / f"{distr}-{version}.dist-info"
        new_di = root / f"{distr}-{new_version}.dist-info"
        old_data = root / f"{distr}-{version}.data"
        if old_di.exists():
            old_di.rename(new_di)
        if old_data.exists():
            old_data.rename(root / f"{distr}-{new_version}.data")

        meta = new_di / "METADATA"
        if meta.exists():
            lines = meta.read_text().splitlines()
            for i, ln in enumerate(lines):
                if ln.startswith("Version: "):
                    lines[i] = f"Version: {new_version}"
                    break
            meta.write_text("\n".join(lines) + "\n")

        # Rebuild RECORD (all files except RECORD carry hash+size; RECORD blank).
        record_path = new_di / "RECORD"
        record_lines = []
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            if rel == record_path.relative_to(root).as_posix():
                continue
            h, sz = _hash_and_size(f.read_bytes())
            record_lines.append(f"{rel},{h},{sz}")
        record_lines.append(f"{record_path.relative_to(root).as_posix()},,")
        record_path.write_text("\n".join(record_lines) + "\n")

        # Re-zip.
        args.out.mkdir(parents=True, exist_ok=True)
        out_whl = args.out / f"{distr}-{new_version}-{rest_tags}.whl"
        with zipfile.ZipFile(out_whl, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(root).as_posix())
        print(f"wrote {out_whl}")


if __name__ == "__main__":
    main()
