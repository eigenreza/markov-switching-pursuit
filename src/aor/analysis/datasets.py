"""Real chases by dataset and split, with caching of the contract tables.

Splits are by animal where animals are identified (Michaiel 2020) and by recording file for
MABe22, where animal identity is not tracked across sequences. Holmgren 2021 is a replication
set with no split. The split membership is fixed here and listed in docs/data.md.

Holmgren's 200 Hz recordings are decimated by a factor of seven (to 28.6 frames per second) so
that one delay grid in frames spans the same range of seconds as for the 30 Hz datasets.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import Chase, ChaseMeta, build_frames, validate_chase
from ..data import DERIVED_DIR, RAW_DIR, holmgren, mabe22, michaiel

SPLITS = {
    "michaiel2020": {
        "dev": ["J470c", "J462c", "J475c", "J463b"],
        "held": ["J462b", "J462a", "J463c"],
    },
    "mabe22": {
        "dev": ["mouse_user_train", "mouse_submission_keypoints"],
        "held": ["mouse_triplet_test"],
    },
    "holmgren2021": {"replication": ["f001", "f002", "f003"]},
}
HOLMGREN_DECIMATION = 7
INFORMATIVE_SPEED = 3.0  # cm/s; frames with the target moving faster than this separate the laws


def cache_path(dataset: str, split: str, variant: str | None = None) -> Path:
    tag = f"{split}_{variant}" if variant else split
    return DERIVED_DIR / dataset / f"chases_{tag}.parquet"


def chases_to_table(chases: list[Chase]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, metas = [], []
    for c in chases:
        df = c.frames.copy()
        df.insert(0, "chase_id", c.meta.chase_id)
        frames.append(df)
        m = c.meta.to_dict()
        m["extra"] = str(m["extra"])
        metas.append(m)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)


def table_to_chases(frames: pd.DataFrame, meta: pd.DataFrame) -> list[Chase]:
    groups = {
        k: g.drop(columns=["chase_id"]).reset_index(drop=True)
        for k, g in frames.groupby("chase_id", sort=False)
    }
    out = []
    for _, m in meta.iterrows():
        cm = ChaseMeta(
            m["dataset"],
            m["session_id"],
            m["animal_id"],
            m["chase_id"],
            float(m["fps"]),
            int(m["start_frame"]),
            int(m["end_frame"]),
            m["end_reason"],
            m["target_type"],
            m["pursuer_point"],
            None if pd.isna(m["px_per_cm"]) else float(m["px_per_cm"]),
            None if pd.isna(m["native_fps"]) else float(m["native_fps"]),
            m["coordinate_frame"],
            None,
            {},
        )
        out.append(Chase(meta=cm, frames=groups[m["chase_id"]]))
    return out


def decimate(chase: Chase, factor: int) -> Chase:
    """Keep every ``factor``-th frame and renumber, keeping the per-frame kinematic columns."""
    f = chase.frames.iloc[::factor].reset_index(drop=True)
    fps = chase.meta.fps / factor
    n = len(f)
    df = build_frames(
        np.arange(n),
        fps,
        f["px"].to_numpy(),
        f["py"].to_numpy(),
        f["p_heading"].to_numpy(),
        f["p_speed"].to_numpy(),
        f["tx"].to_numpy(),
        f["ty"].to_numpy(),
        f["tvx"].to_numpy(),
        f["tvy"].to_numpy(),
        **{
            c: f[c].to_numpy()
            for c in ("t_speed", "t_heading", "px_pix", "py_pix", "tx_pix", "ty_pix")
            if c in f.columns
        },
    )
    meta = replace(
        chase.meta, fps=fps, start_frame=0, end_frame=n - 1, extra={**chase.meta.extra, "decimation": factor}
    )
    return Chase(meta=meta, frames=df)


def _load_raw(dataset: str, split: str, variant: str | None = None) -> list[Chase]:
    if variant is not None and dataset != "michaiel2020":
        raise ValueError(f"no variant {variant!r} for {dataset}")
    if dataset == "michaiel2020":
        chases, _ = michaiel.load_chases(RAW_DIR / "michaiel2020", scale=variant)
        keep = set(SPLITS[dataset][split])
        return [c for c in chases if c.meta.animal_id in keep]
    if dataset == "holmgren2021":
        chases, _ = holmgren.load_chases(RAW_DIR / "holmgren2021", use_author_phases=False)
        out = []
        for c in chases:
            d = decimate(c, HOLMGREN_DECIMATION)
            if len(d.frames) >= 6:
                try:
                    validate_chase(d)
                    out.append(d)
                except Exception:
                    continue
        return out
    if dataset == "mabe22":
        files = [(kf, lf) for kf, lf in mabe22.FILES if kf.replace(".npy", "") in SPLITS[dataset][split]]
        chases, _ = mabe22.load_chases(RAW_DIR / "mabe22", files=files)
        return chases
    raise ValueError(f"unknown dataset {dataset!r}")


def load_real(dataset: str, split: str, refresh: bool = False, variant: str | None = None) -> list[Chase]:
    """Chases of one dataset split, from the cache when present. ``variant`` names an alternative
    loading (the arena-fit scale of the primary dataset) with its own cache."""
    p = cache_path(dataset, split, variant)
    if p.exists() and not refresh:
        frames = pd.read_parquet(p)
        meta = pd.read_parquet(p.with_name(p.name.replace("chases_", "meta_")))
        return table_to_chases(frames, meta)
    chases = _load_raw(dataset, split, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames, meta = chases_to_table(chases)
    frames.to_parquet(p, index=False)
    meta.to_parquet(p.with_name(p.name.replace("chases_", "meta_")), index=False)
    return chases


def informative_mask(chase: Chase, v_min: float = INFORMATIVE_SPEED) -> np.ndarray:
    """Frames at which the target moves faster than ``v_min`` cm/s (smoothed speed)."""
    return chase.frames["t_speed"].to_numpy(dtype=float) > v_min


def summarise_split(chases: list[Chase], v_min: float = INFORMATIVE_SPEED) -> dict:
    n_frames = sum(len(c.frames) for c in chases)
    n_inf = sum(int(informative_mask(c, v_min).sum()) for c in chases)
    animals = sorted({c.meta.animal_id for c in chases})
    return {
        "n_chases": len(chases),
        "n_frames": n_frames,
        "n_informative_frames": n_inf,
        "informative_fraction": n_inf / max(n_frames, 1),
        "n_animals": len(animals),
        "animals": animals,
        "fps": float(chases[0].meta.fps) if chases else None,
        "duration_s_median": float(np.median([c.duration_s for c in chases])) if chases else None,
    }
