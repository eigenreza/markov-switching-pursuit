"""Smoke tests for the recovery study and the figures on a tiny synthetic suite."""

import json

import yaml

from aor import recovery
from aor.figures import fig1_laws, fig2_examples, fig3_recovery
from aor.synthetic import generate

SMALL_SUITE = {
    "base_seed": 5,
    "replicates": 6,
    "min_duration_s": 0.5,
    "defaults": {
        "law": "pure_pursuit",
        "law_params": {"gain": 0.5, "delay": 1, "noise": 0.05, "tau": 0.4, "N": 3.0},
        "target": {"kind": "constant_velocity", "speed": 12.0, "turn_rate_hz": 0.5},
        "world": {"fps": 30.0, "arena": [80.0, 80.0], "capture_radius": 2.0, "max_frames": 120},
    },
    "conditions": [
        {
            "name": "single",
            "group": "single",
            "grid": {"law": ["pure_pursuit", "lead_pursuit"], "law_params.noise": [0.05]},
        },
        {
            "name": "switch_fixed",
            "group": "switching",
            "law": "lead_pursuit",
            "switching": {"kind": "fixed", "fixed": [[0, "lead_pursuit"], [45, "pure_pursuit"]]},
        },
        {"name": "tau", "group": "parameters", "law": "lead_pursuit", "grid": {"law_params.tau": [0.3]}},
    ],
}

SMALL_MODEL = {
    "inference": {
        "delays": [0, 1, 2],
        "tau_bounds": [0.0, 1.5],
        "n_iter": 15,
        "tol": 1.0e-3,
        "n_restarts": 1,
        "profile_n_iter": 8,
        "profile_n_restarts": 1,
        "l2": 1.0e-2,
        "manoeuvre": {
            "lag_frames": 3,
            "heading_change_deg": 60.0,
            "speed_change_cm_s": 15.0,
            "min_speed_cm_s": 3.0,
            "recent_frames": 6,
        },
    },
    "recovery": {
        "lengths_frames": [30, 60],
        "detection_level": 0.95,
        "bootstrap_B": 4,
        "calibration_bins": 5,
        "accuracy_target": 0.9,
        "detection_target": 0.8,
        "false_detection_max": 0.1,
        "example_noise": 0.05,
        "seed": 0,
    },
}


def test_recovery_and_figures_end_to_end(tmp_path):
    suite_cfg = tmp_path / "synthetic.yaml"
    suite_cfg.write_text(yaml.safe_dump(SMALL_SUITE), encoding="utf-8")
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(yaml.safe_dump(SMALL_MODEL), encoding="utf-8")
    suite_dir = tmp_path / "suite"
    generate(suite_cfg, suite_dir)
    rec_dir = tmp_path / "recovery"
    summary = recovery.run(rec_dir, suite_dir, model_cfg, verbose=False)
    for name in (
        "classification.csv",
        "detection.csv",
        "switch_time.csv",
        "params.csv",
        "calibration.csv",
        "summary.json",
        "examples.json",
        "delay_profiles.csv",
    ):
        assert (rec_dir / name).exists(), name
    assert "limits" in summary and "classification_accuracy" in summary
    ex = json.loads((rec_dir / "examples.json").read_text(encoding="utf-8"))
    assert set(ex) <= {"pure_pursuit", "lead_pursuit", "proportional_navigation"}
    fig_dir = tmp_path / "figs"
    out1 = fig1_laws.make(fig_dir)
    assert all(p.exists() for p in out1)
    if ex:
        out2 = fig2_examples.make(rec_dir / "examples.json", fig_dir)
        assert all(p.exists() for p in out2)
    out3 = fig3_recovery.make(rec_dir, fig_dir)
    assert all(p.exists() for p in out3)
    # no creation dates in the PDF metadata
    pdf = (fig_dir / "fig1_laws.pdf").read_bytes()
    assert b"CreationDate" not in pdf and b"ModDate" not in pdf
