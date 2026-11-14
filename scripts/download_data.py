"""Download and extract the NCT-CRC-HE histology datasets from Zenodo.

File names, sizes and md5 checksums are read from the Zenodo API, so the
script stays correct if the record's download links change.

Features
- Resumes interrupted downloads (keeps a .part file and uses HTTP Range).
- Verifies each archive against Zenodo's md5 checksum.
- Skips datasets that were already extracted (marker file).

Usage
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py 'datasets=[CRC-VAL-HE-7K]'   # small test first
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path

import hydra
import requests
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_FILE = "https://zenodo.org/records/{record_id}/files/{key}?download=1"


def fetch_file_index(record_id: int, timeout: int) -> dict[str, dict]:
    """Return {file name: metadata} for every file in a Zenodo record."""
    response = requests.get(ZENODO_API.format(record_id=record_id), timeout=timeout)
    response.raise_for_status()
    return {entry["key"]: entry for entry in response.json()["files"]}


def md5sum(path: Path, chunk_size: int) -> str:
    """Compute the md5 hex digest of a file, reading it in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as handle, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True, desc=f"md5 {path.name}"
    ) as bar:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
            bar.update(len(block))
    return digest.hexdigest()


def download(url: str, dest: Path, expected_size: int, timeout: int, chunk_size: int) -> None:
    """Stream a file to dest, resuming from dest.part if it exists."""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    if expected_size and done >= expected_size:
        part.rename(dest)
        return

    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if done and response.status_code != 206:
            log.warning("Server ignored resume request; restarting %s", dest.name)
            done = 0

        mode = "ab" if done else "wb"
        with open(part, mode) as handle, tqdm(
            total=expected_size or None,
            initial=done,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for block in response.iter_content(chunk_size=chunk_size):
                handle.write(block)
                bar.update(len(block))

    part.rename(dest)


def extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive into out_dir with a progress bar."""
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            archive.extract(member, out_dir)


@hydra.main(version_base="1.3", config_path="../conf", config_name="download")
def main(cfg: DictConfig) -> None:
    raw_dir = Path(to_absolute_path(cfg.paths.raw_dir))
    data_dir = Path(to_absolute_path(cfg.paths.data_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    index = fetch_file_index(cfg.zenodo.record_id, cfg.download.timeout)
    log.info("Zenodo record %s contains: %s", cfg.zenodo.record_id, sorted(index))

    for name in cfg.datasets:
        key = f"{name}.zip"
        if key not in index:
            raise KeyError(f"{key} not found in record; available: {sorted(index)}")

        marker = data_dir / f".{name}.extracted"
        if marker.exists():
            log.info("%s already extracted, skipping", name)
            continue

        meta = index[key]
        zip_path = raw_dir / key
        url = meta.get("links", {}).get("self") or ZENODO_FILE.format(
            record_id=cfg.zenodo.record_id, key=key
        )

        if not zip_path.exists():
            log.info("Downloading %s (%.1f GB)", key, meta.get("size", 0) / 1e9)
            download(url, zip_path, meta.get("size", 0), cfg.download.timeout, cfg.download.chunk_size)

        if cfg.download.verify_md5:
            expected = meta["checksum"].split(":", 1)[-1]
            actual = md5sum(zip_path, cfg.download.chunk_size)
            if actual != expected:
                zip_path.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for {key} (expected {expected}, got {actual}); "
                    "the corrupt file was deleted, run the script again."
                )
            log.info("%s checksum OK", key)

        extract(zip_path, data_dir)
        marker.touch()
        log.info("%s extracted to %s", name, data_dir / name)

        if not cfg.download.keep_zip:
            zip_path.unlink()

    log.info("Done.")


if __name__ == "__main__":
    main()
