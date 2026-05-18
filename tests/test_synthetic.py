import json

import yaml

from aor.contracts import validate_chase
from aor.synthetic import chases_from_suite, condition_seed, expand_conditions, generate, load_suite

SMALL = {
    "base_seed": 3,
    "replicates": 2,
    "min_duration_s": 0.3,
    "defaults": {
        "law": "pure_pursuit",
        "law_params": {"gain": 0.5, "delay": 1, "noise": 0.05, "tau": 0.4, "N": 3.0},
        "target": {"kind": "constant_velocity", "speed": 8.0, "turn_rate_hz": 0.5},
        "world": {"fps": 30.0, "arena": [45.0, 38.0], "capture_radius": 2.0, "max_frames": 90},
    },
    "conditions": [
        {
            "name": "single",
            "group": "single",
            "grid": {"law": ["pure_pursuit", "lead_pursuit"], "law_params.noise": [0.05, 0.1]},
        },
        {
            "name": "sw",
            "group": "switching",
            "law": "lead_pursuit",
            "switching": {"kind": "fixed", "fixed": [[0, "lead_pursuit"], [8, "pure_pursuit"]]},
        },
        {
            "name": "jump",
            "group": "misspecified",
            "target": {"kind": "jumper", "jump_rate_hz": 1.0},
            "speed_mean": 10.0,
        },
    ],
}


def write_cfg(tmp_path):
    p = tmp_path / "synthetic.yaml"
    p.write_text(yaml.safe_dump(SMALL), encoding="utf-8")
    return p


def test_expand_conditions_builds_grid_names():
    conds = expand_conditions(SMALL)
    names = [c["name"] for c in conds]
    assert len(conds) == 4 + 1 + 1
    assert "single__law=lead_pursuit__law_params.noise=0.1" in names
    lead = next(c for c in conds if c["name"] == "single__law=lead_pursuit__law_params.noise=0.1")
    assert lead["law"] == "lead_pursuit" and lead["law_params"]["noise"] == 0.1


def test_condition_seed_is_stable_and_distinct():
    assert condition_seed("a", 0, 1) == condition_seed("a", 0, 1)
    assert condition_seed("a", 0, 1) != condition_seed("a", 1, 1)
    assert condition_seed("a", 0, 1) != condition_seed("b", 0, 1)


def test_generate_is_deterministic_and_round_trips(tmp_path):
    cfg = write_cfg(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    s1 = generate(cfg, out1)
    s2 = generate(cfg, out2)
    assert s1["frames_sha256"] == s2["frames_sha256"]
    assert s1["n_chases"] >= 10
    frames, meta = load_suite(out1)
    chases = chases_from_suite(frames, meta)
    assert len(chases) == s1["n_chases"]
    for c in chases:
        validate_chase(c)
        assert "law_true" in c.frames.columns
    man = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
    assert man["n_chases"] == s1["n_chases"]
    sw = meta[meta["condition"] == "sw"]
    assert (sw["n_switches"] >= 1).all()
    assert set(meta["animal_id"]) == {"single", "switching", "misspecified"}
