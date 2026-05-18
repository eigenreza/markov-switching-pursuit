"""Download and verify the raw datasets listed in configs/data.yaml and configs/manifests/.

Every file is fetched with curl into data/raw/<dataset>/ and checked against the hash recorded
in the configuration. Files that already exist with the right hash are skipped. Nothing here
writes outside data/raw.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from . import RAW_DIR, load_config

ZENODO = "https://zenodo.org/api/records/{record}/files/{name}/content"
CALTECH = "https://data.caltech.edu/api/records/{record}/files/{name}/content"

SOURCES = {
    "michaiel2020": [
        (
            "Michaiel_et_al.2020_fullDataset.mat",
            ZENODO.format(record=3975832, name="Michaiel_et_al.2020_fullDataset.mat"),
        ),
        (
            "compileAllAnimals_CONTROL_090219.mat",
            ZENODO.format(record=3975832, name="compileAllAnimals_CONTROL_090219.mat"),
        ),
        ("readme.txt", ZENODO.format(record=3975832, name="readme.txt")),
    ],
    "holmgren2021": [
        (n, ZENODO.format(record=5576696, name=n))
        for n in ("README.txt", "behavioral_classifications.zip", "meshes.zip", "trajectories.zip")
    ],
    "mabe22": [
        (n, CALTECH.format(record="rdsa8-rde65", name=n))
        for n in (
            "mouse_frame_number_map.npy",
            "mouse_user_train.npy",
            "mouse_triplets_test_labels.npy",
            "mouse_submission_labels.npy",
            "mouse_submission_keypoints.npy",
            "mouse_triplet_test.npy",
        )
    ],
}


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        r = subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(dest), url], check=False)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return
    raise RuntimeError(f"download failed: {url}")


def expected_hashes(dataset: str) -> dict[str, tuple[str, str]]:
    cfg = load_config()["datasets"][dataset]
    out = {}
    for f in cfg.get("files", []):
        if "md5" in f:
            out[f["name"]] = ("md5", f["md5"])
    return out


def download_dataset(dataset: str, verify_only: bool = False) -> list[str]:
    """Return a list of status lines."""
    lines = []
    want = expected_hashes(dataset)
    for name, url in SOURCES[dataset]:
        dest = RAW_DIR / dataset / name
        algo, digest = want.get(name, (None, None))
        if dest.exists() and digest and file_hash(dest, algo) == digest:
            lines.append(f"ok       {dataset}/{name}")
            continue
        if verify_only:
            lines.append(f"missing  {dataset}/{name}")
            continue
        fetch(url, dest)
        status = "ok" if (digest is None or file_hash(dest, algo) == digest) else "BAD-HASH"
        lines.append(f"{status:8s} {dataset}/{name}")
    return lines
