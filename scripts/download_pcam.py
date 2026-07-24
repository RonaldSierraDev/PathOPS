#!/usr/bin/env python3
"""Download and verify the PatchCamelyon (PCam) dataset from the Zenodo mirror."""
import argparse
import gzip
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ZENODO_CONTENT_URL = "https://zenodo.org/api/records/2546921/files/{key}/content"

# Official checksums from https://github.com/basveeling/pcam (source of truth).
# `source` is the filename Zenodo hosts it under. On Zenodo, the valid/ and test/
# files are swapped relative to these official checksums (train is unaffected) --
# confirmed by checksum comparison, not assumed. `source` corrects for that so the
# file saved locally as e.g. valid_x is actually the official valid split.
FILES = [
    dict(split="train", kind="x", name="camelyonpatch_level_2_split_train_x.h5.gz",
         source="camelyonpatch_level_2_split_train_x.h5.gz", md5="1571f514728f59376b705fc836ff4b63"),
    dict(split="train", kind="y", name="camelyonpatch_level_2_split_train_y.h5.gz",
         source="camelyonpatch_level_2_split_train_y.h5.gz", md5="35c2d7259d906cfc8143347bb8e05be7"),
    dict(split="train", kind="meta", name="camelyonpatch_level_2_split_train_meta.csv",
         source="camelyonpatch_level_2_split_train_meta.csv", md5="5a3dd671e465cfd74b5b822125e65b0a"),
    dict(split="valid", kind="x", name="camelyonpatch_level_2_split_valid_x.h5.gz",
         source="camelyonpatch_level_2_split_test_x.h5.gz", md5="d8c2d60d490dbd479f8199bdfa0cf6ec"),
    dict(split="valid", kind="y", name="camelyonpatch_level_2_split_valid_y.h5.gz",
         source="camelyonpatch_level_2_split_test_y.h5.gz", md5="60a7035772fbdb7f34eb86d4420cf66a"),
    dict(split="valid", kind="meta", name="camelyonpatch_level_2_split_valid_meta.csv",
         source="camelyonpatch_level_2_split_test_meta.csv", md5="3455fd69135b66734e1008f3af684566"),
    dict(split="test", kind="x", name="camelyonpatch_level_2_split_test_x.h5.gz",
         source="camelyonpatch_level_2_split_valid_x.h5.gz", md5="d5b63470df7cfa627aeec8b9dc0c066e"),
    dict(split="test", kind="y", name="camelyonpatch_level_2_split_test_y.h5.gz",
         source="camelyonpatch_level_2_split_valid_y.h5.gz", md5="2b85f58b927af9964a4c15b8f7e8f179"),
    dict(split="test", kind="meta", name="camelyonpatch_level_2_split_test_meta.csv",
         source="camelyonpatch_level_2_split_valid_meta.csv", md5="67589e00a4a37ec317f2d1932c7502ca"),
]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = read / total * 100
                print(f"\r  {dest.name}: {pct:5.1f}% ({read / 1e6:.0f} / {total / 1e6:.0f} MB)", end="")
    print()
    tmp.rename(dest)


def fetch_file(entry: dict, data_dir: Path, force: bool) -> Path:
    dest = data_dir / entry["name"]
    if dest.exists() and not force:
        if md5sum(dest) == entry["md5"]:
            print(f"[skip] {entry['name']} already present and verified")
            return dest
        print(f"[warn] {entry['name']} exists but checksum mismatch, re-downloading")

    url = ZENODO_CONTENT_URL.format(key=entry["source"])
    print(f"[get]  {entry['name']}  (from Zenodo as {entry['source']})")
    download(url, dest)

    actual = md5sum(dest)
    if actual != entry["md5"]:
        dest.unlink()
        raise SystemExit(
            f"Checksum FAILED for {entry['name']}: expected {entry['md5']}, got {actual}"
        )
    print(f"[ok]   {entry['name']} checksum verified")
    return dest


def decompress(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")  # strip .gz
    if out_path.exists():
        print(f"[skip] {out_path.name} already decompressed")
        return out_path
    print(f"[dec]  {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam", help="Where to store the dataset")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"],
                         default=["train", "valid", "test"], help="Which splits to fetch")
    parser.add_argument("--force", action="store_true", help="Re-download even if verified files exist")
    parser.add_argument("--keep-gz", action="store_true", help="Keep .h5.gz files after decompressing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in FILES if e["split"] in args.splits]
    for entry in entries:
        path = fetch_file(entry, data_dir, args.force)
        if entry["kind"] in ("x", "y"):
            decompress(path)
            if not args.keep_gz:
                path.unlink()

    print(f"\nDone. Dataset ready in {data_dir.resolve()}")


if __name__ == "__main__":
    main()
