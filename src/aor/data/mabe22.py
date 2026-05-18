"""Loader for the MABe22 mouse triplets (Caltech Data record rdsa8-rde65).

Each sequence is one minute of three mice at 30 frames per second with twelve keypoints per
mouse in pixels. Missing markers are stored as (0, 0). The keypoint order was established from
inter-point distances: nose, left ear, right ear, neck, left fore paw, right fore paw, centre
of the back, left hind paw, right hind paw, tail base, tail middle, tail tip. The files are not
all in the same pixel frame, so the scale is estimated per file from the extent of the centre
of the back against the 52 cm arena.

Chases are derived for every ordered pair of mice with the shared kinematic rule and kept when
the target itself is moving. The frame-level "chases" label, where present, serves as a check
on the rule, not as the definition of a chase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Chase, ChaseMeta, ContractError, build_frames, validate_chase
from ..kinematics import heading_from_axis, interpolate_gaps_xy, runs_of_true, velocity
from ..kinematics import speed as speed_of
from ..segment import SegmentationParams, segment_chases

DATASET = "mabe22"
FPS = 30.0
ARENA_CM = 52.0
KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_fore_paw",
    "right_fore_paw",
    "centre_back",
    "left_hind_paw",
    "right_hind_paw",
    "tail_base",
    "tail_middle",
    "tail_tip",
]
NOSE, NECK, CENTRE, TAILBASE = 0, 3, 6, 9
GAP_FRAMES = 5
MIN_TARGET_SPEED = 5.0  # cm/s mean over the episode, for the target to count as moving
FILES = [
    ("mouse_user_train.npy", None),
    ("mouse_submission_keypoints.npy", "mouse_submission_labels.npy"),
    ("mouse_triplet_test.npy", "mouse_triplets_test_labels.npy"),
]


def load_sequences(path: Path) -> tuple[dict, list | None]:
    d = np.load(path, allow_pickle=True).item()
    return d["sequences"], d.get("vocabulary")


def load_labels(path: Path) -> tuple[dict, np.ndarray] | None:
    if path is None or not Path(path).exists():
        return None
    L = np.load(path, allow_pickle=True).item()
    if "chases" not in L["vocabulary"]:
        return None
    return L["frame_number_map"], L["label_array"][L["vocabulary"].index("chases")]


def clean_keypoints(kp: np.ndarray) -> np.ndarray:
    kp = kp.astype(float)
    kp[(kp[..., 0] == 0) & (kp[..., 1] == 0)] = np.nan
    kp[kp > 2000] = np.nan
    return kp


def estimate_px_per_cm(seqs: dict) -> float:
    pts = np.concatenate(
        [clean_keypoints(s["keypoints"])[:, :, CENTRE, :].reshape(-1, 2) for s in seqs.values()]
    )
    lo = np.nanpercentile(pts, 0.05, axis=0)
    hi = np.nanpercentile(pts, 99.95, axis=0)
    return float(np.mean(hi - lo) / ARENA_CM)


def sequence_chases(
    sid: str, seq: dict, px_per_cm: float, params: SegmentationParams, file_tag: str
) -> tuple[list[Chase], list[str], np.ndarray]:
    kp = clean_keypoints(seq["keypoints"]) / px_per_cm
    n = kp.shape[0]
    cent = [interpolate_gaps_xy(kp[:, m, CENTRE, :], GAP_FRAMES) for m in range(3)]
    head = [
        heading_from_axis(
            interpolate_gaps_xy(kp[:, m, TAILBASE, :], GAP_FRAMES),
            interpolate_gaps_xy(kp[:, m, NECK, :], GAP_FRAMES),
            window=3,
        )
        for m in range(3)
    ]
    spd = [speed_of(c, FPS, window=3) for c in cent]
    vel = [velocity(c, FPS, window=5) for c in cent]
    chases, rejections = [], []
    mask = np.zeros(n, bool)
    for p in range(3):
        for q in range(3):
            if p == q:
                continue
            for g in segment_chases(
                cent[p][:, 0], cent[p][:, 1], head[p], cent[q][:, 0], cent[q][:, 1], FPS, params
            ):
                sl = slice(g.start, g.end)
                if not (np.nanmean(spd[q][sl]) > MIN_TARGET_SPEED):
                    continue
                df = build_frames(
                    np.arange(g.start, g.end),
                    FPS,
                    cent[p][sl, 0],
                    cent[p][sl, 1],
                    head[p][sl],
                    spd[p][sl],
                    cent[q][sl, 0],
                    cent[q][sl, 1],
                    vel[q][sl, 0],
                    vel[q][sl, 1],
                    t_heading=head[q][sl],
                    t_speed=spd[q][sl],
                    px_pix=cent[p][sl, 0] * px_per_cm,
                    py_pix=cent[p][sl, 1] * px_per_cm,
                    tx_pix=cent[q][sl, 0] * px_per_cm,
                    ty_pix=cent[q][sl, 1] * px_per_cm,
                )
                meta = ChaseMeta(
                    DATASET,
                    f"{file_tag}:{sid}",
                    f"{file_tag}:{sid}:m{p}",
                    f"{sid}_m{p}m{q}_{g.start:04d}",
                    FPS,
                    int(g.start),
                    int(g.end - 1),
                    g.end_reason,
                    "mouse",
                    pursuer_point="centre_back",
                    px_per_cm=px_per_cm,
                    native_fps=FPS,
                    coordinate_frame="overhead_camera_px",
                    extra={"pursuer_index": p, "target_index": q, "file": file_tag},
                )
                chase = Chase(meta=meta, frames=df)
                try:
                    validate_chase(chase, min_duration_s=params.min_duration_s)
                except ContractError as e:
                    rejections.append(str(e).splitlines()[1].strip() if "\n" in str(e) else str(e))
                    continue
                chases.append(chase)
                mask[g.start : g.end] = True
    return chases, rejections, mask


def audit_file(
    raw_dir: Path,
    keypoints_file: str,
    labels_file: str | None,
    params: SegmentationParams,
    max_sequences: int | None = None,
) -> tuple[list[Chase], dict]:
    seqs, vocab = load_sequences(Path(raw_dir) / keypoints_file)
    labels = load_labels(Path(raw_dir) / labels_file) if labels_file else None
    px_per_cm = estimate_px_per_cm(seqs)
    tag = keypoints_file.replace(".npy", "")
    chases, n_rej, frames = [], 0, 0
    ann_bouts, ann_hit, ann_lens = 0, 0, []
    body = []
    for i, (sid, s) in enumerate(seqs.items()):
        if max_sequences is not None and i >= max_sequences:
            break
        c, r, mask = sequence_chases(sid, s, px_per_cm, params, tag)
        chases.extend(c)
        n_rej += len(r)
        frames += s["keypoints"].shape[0]
        kp = clean_keypoints(s["keypoints"]) / px_per_cm
        body.append(np.nanmedian(np.linalg.norm(kp[:, :, NOSE] - kp[:, :, TAILBASE], axis=-1)))
        lab = None
        if labels is not None and sid in labels[0]:
            a0, a1 = labels[0][sid]
            lab = labels[1][a0:a1] > 0.5
        elif "annotations" in s and vocab and "chases" in vocab:
            lab = s["annotations"][vocab.index("chases")] > 0.5
        if lab is not None:
            st, en = runs_of_true(lab)
            for a, b in zip(st, en, strict=True):
                ann_bouts += 1
                ann_lens.append(b - a)
                ann_hit += int(mask[a:b].mean() > 0.3)
    info = {
        "file": keypoints_file,
        "n_sequences": len(seqs),
        "hours": frames / FPS / 3600,
        "px_per_cm": px_per_cm,
        "median_nose_to_tail_base_cm": float(np.nanmedian(body)),
        "n_rejected_by_contract": n_rej,
        "labelled_chase_bouts": ann_bouts,
        "labelled_bout_frames_median": float(np.median(ann_lens)) if ann_lens else None,
        "labelled_bout_frames_max": int(max(ann_lens)) if ann_lens else None,
        "labelled_bouts_recovered_by_rule": ann_hit,
    }
    del seqs
    return chases, info


def load_chases(
    raw_dir: Path, params: SegmentationParams = SegmentationParams(), files=FILES
) -> tuple[list[Chase], dict]:
    chases, infos = [], {}
    for kf, lf in files:
        if not (Path(raw_dir) / kf).exists():
            continue
        c, info = audit_file(raw_dir, kf, lf, params)
        chases.extend(c)
        infos[kf] = info
    return chases, infos


def audit(raw_dir: Path, params: SegmentationParams = SegmentationParams()) -> dict:
    chases, infos = load_chases(raw_dir, params)
    dur = np.array([c.duration_s for c in chases])
    return {
        "dataset": DATASET,
        "fps": FPS,
        "arena_cm": ARENA_CM,
        "keypoints": KEYPOINTS,
        "licence": "CC-BY-NC-SA (dataset paper datasheet)",
        "files": infos,
        "n_sequences": int(sum(i["n_sequences"] for i in infos.values())),
        "hours": float(sum(i["hours"] for i in infos.values())),
        "labelled_chase_bouts": int(sum(i["labelled_chase_bouts"] for i in infos.values())),
        "labelled_bouts_recovered_by_rule": int(
            sum(i["labelled_bouts_recovered_by_rule"] for i in infos.values())
        ),
        "segmentation": {**params.__dict__, "min_target_speed_mean": MIN_TARGET_SPEED},
        "n_chases": len(chases),
        "n_rejected_by_contract": int(sum(i["n_rejected_by_contract"] for i in infos.values())),
        "chase_duration_s": {
            "median": float(np.median(dur)),
            "q25": float(np.percentile(dur, 25)),
            "q75": float(np.percentile(dur, 75)),
            "max": float(dur.max()),
        }
        if len(dur)
        else None,
        "n_chases_ge_1s": int((dur >= 1).sum()),
        "n_chases_ge_2s": int((dur >= 2).sum()),
        "n_chases_ge_3s": int((dur >= 3).sum()),
        "end_reasons": {
            r: int(sum(c.meta.end_reason == r for c in chases))
            for r in sorted({c.meta.end_reason for c in chases})
        },
    }
