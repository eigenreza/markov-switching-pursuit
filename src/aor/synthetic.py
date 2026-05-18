"""The synthetic suite: a fixed, seeded set of simulated chases for the recovery study.

The suite is described by ``configs/synthetic.yaml``. Every chase has a deterministic seed
derived from its condition and replicate index, so the suite regenerates bit for bit. The
chases are written as one parquet table of frames and one of metadata, with the true law per
frame in the frames table and the generating parameters in the metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contracts import Chase, validate_chase
from .laws import LawParams
from .simulator import PursuerParams, SwitchParams, TargetParams, WorldParams, simulate, to_chase

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "synthetic.yaml"
OUT_DIR = ROOT / "data" / "derived" / "synthetic"


def condition_seed(name: str, rep: int, base: int) -> int:
    h = hashlib.sha256(f"{base}:{name}:{rep}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _law_params(d: dict) -> LawParams:
    return LawParams(**{k: d[k] for k in ("gain", "delay", "noise", "tau", "N", "noise_ar") if k in d})


def build_condition(
    cond: dict, defaults: dict
) -> tuple[TargetParams, PursuerParams, SwitchParams, WorldParams]:
    target = TargetParams(**{**defaults.get("target", {}), **cond.get("target", {})})
    lp = {**defaults.get("law_params", {}), **cond.get("law_params", {})}
    per_law = {k: _law_params({**lp, **v}) for k, v in cond.get("law_params_by_law", {}).items()}
    pursuer = PursuerParams(
        law=cond.get("law", defaults.get("law", "pure_pursuit")),
        params=_law_params(lp),
        speed_mean=cond.get("speed_mean", defaults.get("speed_mean", 20.0)),
        speed_ar=cond.get("speed_ar", defaults.get("speed_ar", 0.9)),
        speed_sd=cond.get("speed_sd", defaults.get("speed_sd", 2.0)),
        law_params=per_law or None,
    )
    sw = cond.get("switching", {})
    if "fixed" in sw:
        sw = {**sw, "fixed": tuple((int(f), str(law)) for f, law in sw["fixed"])}
    if "transition" in sw:
        sw = {**sw, "transition": tuple(tuple(float(x) for x in row) for row in sw["transition"])}
    switching = SwitchParams(**sw)
    world = WorldParams(**{**defaults.get("world", {}), **cond.get("world", {})})
    return target, pursuer, switching, world


def expand_conditions(cfg: dict) -> list[dict]:
    """Expand grid entries into flat conditions. A grid entry has ``grid`` with lists; every
    combination becomes a condition named ``<name>__k=v__...``."""
    out = []
    for c in cfg["conditions"]:
        grid = c.get("grid")
        if not grid:
            out.append(c)
            continue
        keys = list(grid)
        import itertools

        for combo in itertools.product(*[grid[k] for k in keys]):
            cc = json.loads(json.dumps({k: v for k, v in c.items() if k != "grid"}))
            parts = []
            for k, v in zip(keys, combo, strict=True):
                parts.append(f"{k}={v}")
                target = cc
                path = k.split(".")
                for pk in path[:-1]:
                    target = target.setdefault(pk, {})
                target[path[-1]] = v
            cc["name"] = c["name"] + "__" + "__".join(parts)
            out.append(cc)
    return out


def generate(config_path: Path = CONFIG, out_dir: Path = OUT_DIR) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    base_seed = int(cfg.get("base_seed", 0))
    conditions = expand_conditions(cfg)
    frames_tables, metas = [], []
    n_ok = n_short = 0
    for cond in conditions:
        target, pursuer, switching, world = build_condition(cond, defaults)
        reps = int(cond.get("replicates", cfg.get("replicates", 10)))
        for rep in range(reps):
            seed = condition_seed(cond["name"], rep, base_seed)
            res = simulate(target, pursuer, switching, world, seed=seed)
            chase_id = f"{cond['name']}__r{rep:02d}"
            chase = to_chase(
                res,
                chase_id,
                session_id=cond["name"],
                animal_id=cond.get("group", cond["name"].split("__")[0]),
            )
            try:
                validate_chase(chase, min_duration_s=float(cfg.get("min_duration_s", 0.5)))
            except Exception:
                n_short += 1
                continue
            df = chase.frames.copy()
            df.insert(0, "chase_id", chase_id)
            frames_tables.append(df)
            m = chase.meta.to_dict()
            m.update(
                {
                    "n_frames": len(df),
                    "condition": cond["name"],
                    "replicate": rep,
                    "manoeuvre_frames": int(res.manoeuvre.sum()),
                    "n_switches": int((np.diff(res.law_index) != 0).sum()),
                    "law_true_first": int(res.law_index[0]),
                    "law": pursuer.law,
                    "target_kind": target.kind,
                    "switching_kind": switching.kind,
                    "gain": pursuer.params.gain,
                    "delay": pursuer.params.delay,
                    "noise": pursuer.params.noise,
                    "tau": pursuer.params.tau,
                    "N": pursuer.params.N,
                    "max_frames": world.max_frames,
                }
            )
            m["extra"] = json.dumps(m["extra"], default=str)
            metas.append(m)
            n_ok += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = pd.concat(frames_tables, ignore_index=True)
    meta = pd.DataFrame(metas)
    frames.to_parquet(out_dir / "frames.parquet", index=False)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    summary = {
        "n_conditions": len(conditions),
        "n_chases": n_ok,
        "n_rejected_short": n_short,
        "n_frames": len(frames),
        "base_seed": base_seed,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "frames_sha256": hashlib.sha256((out_dir / "frames.parquet").read_bytes()).hexdigest(),
        "chases_per_group": meta.groupby("animal_id").size().to_dict(),
        "duration_s": {
            "median": float(meta["n_frames"].median() / 30.0),
            "min": float(meta["n_frames"].min() / 30.0),
            "max": float(meta["n_frames"].max() / 30.0),
        },
        "end_reasons": meta["end_reason"].value_counts().to_dict(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def load_suite(out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(out_dir / "frames.parquet"), pd.read_parquet(out_dir / "meta.parquet")


def chases_from_suite(frames: pd.DataFrame, meta: pd.DataFrame) -> list[Chase]:
    """Rebuild Chase objects from the two tables."""
    from .contracts import ChaseMeta

    out = []
    groups = {
        k: g.drop(columns=["chase_id"]).reset_index(drop=True)
        for k, g in frames.groupby("chase_id", sort=False)
    }
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
            float(m["native_fps"]),
            m["coordinate_frame"],
            None,
            json.loads(m["extra"]) if isinstance(m["extra"], str) else {},
        )
        out.append(Chase(meta=cm, frames=groups[m["chase_id"]]))
    return out
