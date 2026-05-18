"""Interactive demo: a live simulator with inference, and a browser of the real chases.

Run from the repository root with `python tasks.py demo` or
`streamlit run demo/app.py`. Everything is computed locally; the results browser reads only
committed result files, and the playground fits one short simulated chase on the fly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aor.figures.style import GREY, LAW_COLOURS, apply_style  # noqa: E402
from aor.inference.models import FitSettings, fit_switching, posteriors, state_roles  # noqa: E402
from aor.laws import LAWS, LawParams  # noqa: E402
from aor.simulator import PursuerParams, SwitchParams, TargetParams, WorldParams, simulate, to_chase  # noqa: E402

RESULTS = ROOT / "results"
STATE_COLOURS = [LAW_COLOURS[law] for law in LAWS] + ["#999999"]
STATE_NAMES = ["pure pursuit", "lead pursuit", "proportional navigation", "persistence"]
LABELS = {
    "michaiel2020": "mouse and cricket (primary)",
    "holmgren2021": "mouse and cricket (replication)",
    "mabe22": "mouse and mouse",
}
REPO = "https://github.com/eigenreza/markov-switching-pursuit"

st.set_page_config(page_title="Anticipate or react? Markov-switching pursuit", layout="wide")
apply_style()


def draw_chase(px, py, tx, ty, path, upto: int | None = None, title: str = "", roles=None):
    """Track figure: target in black, pursuer coloured by the decoded state."""
    n = len(px) if upto is None else int(upto)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(tx[:n], ty[:n], "-", color="#000000", lw=1.2, label="target")
    start = 0
    for j in range(1, n):
        if path[j] != path[start] or j == n - 1:
            end = j if path[j] != path[start] else j + 1
            ax.plot(px[start:end], py[start:end], "-", color=STATE_COLOURS[int(path[start])], lw=2.2)
            start = j
    if n:
        ax.plot([px[n - 1]], [py[n - 1]], "s", color=GREY, ms=7)
        ax.plot([tx[n - 1]], [ty[n - 1]], "o", color="#000000", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(title, fontsize=9)
    for k, name in enumerate(STATE_NAMES[: max(int(np.max(path)) + 1, 3)]):
        label = f"{name} ({roles[k]})" if roles is not None and k < len(roles) else name
        ax.plot([], [], color=STATE_COLOURS[k], lw=3, label=label)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def draw_posterior(t, probs, upto: int | None = None, truth=None):
    n = len(t) if upto is None else int(upto)
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    bottom = np.zeros(n)
    for k, pr in enumerate(probs):
        ax.fill_between(t[:n], bottom, bottom + pr[:n], color=STATE_COLOURS[k], alpha=0.85, lw=0)
        bottom = bottom + pr[:n]
    if truth is not None:
        ax.plot(
            t[:n],
            (np.asarray(truth[:n]) == 0).astype(float) * 0.9 + 0.05,
            "k--",
            lw=0.8,
            label="true law (high = pure)",
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("posterior")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- playground


@st.cache_data(show_spinner=False)
def run_playground(target_kind, law, tau, N, gain, delay, noise, switch, seed):
    params = LawParams(gain=gain, delay=int(delay), noise=noise, tau=tau, N=N)
    sw = (
        SwitchParams(kind="fixed", fixed=((60, "pure_pursuit"),), base_law=law)
        if switch and law != "pure_pursuit"
        else SwitchParams()
    )
    res = simulate(
        TargetParams(kind=target_kind, speed=10.0, turn_rate_hz=0.6, jump_rate_hz=0.6),
        PursuerParams(law=law, params=params, speed_mean=16.0),
        sw,
        WorldParams(arena=(80.0, 80.0), max_frames=150, capture_radius=1.5),
        seed=int(seed),
    )
    chase = to_chase(res, "playground", animal_id="sim")
    if len(chase.frames) < 12:
        return None
    fit = fit_switching([chase], FitSettings(delay=int(delay), n_iter=30, n_restarts=1))
    post = posteriors([chase], fit)[0]
    f = chase.frames
    return {
        "px": f["px"].to_numpy(),
        "py": f["py"].to_numpy(),
        "tx": f["tx"].to_numpy(),
        "ty": f["ty"].to_numpy(),
        "t": f["t"].to_numpy()[post["s"]],
        "s": post["s"],
        "gamma": post["gamma"],
        "path": post["path"],
        "law_true": post["law_true"],
        "fit": {
            "gain": fit.emissions.gain,
            "tau": fit.emissions.tau,
            "N": fit.emissions.N,
            "sigma": fit.emissions.sigma,
            "roles": state_roles(fit),
        },
        "n_frames": len(f),
    }


def page_playground():
    st.header("Playground: build a chase and see what the model makes of it")
    st.write(
        "Set up a chase of your own. Choose how the target moves, which rule the pursuer steers by, and "
        "how quick, delayed and noisy its steering is. The model is then fitted to this one chase without "
        "being told the rule. The coloured strip under the track shows, moment by moment, how likely it "
        "thinks each rule is, and the dashed line marks what the pursuer was really doing. Tick the switch "
        "box to make the pursuer change to pure pursuit partway through, and see whether the model notices."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        target_kind = st.selectbox("target", ["constant_velocity", "evader", "jumper"], index=0)
        law = st.selectbox("pursuer law", LAWS, index=1, format_func=lambda x: x.replace("_", " "))
        tau = st.slider("lead time tau (s)", 0.0, 1.5, 0.5, 0.05)
        N = st.slider("navigation constant N", 0.5, 6.0, 3.0, 0.5)
        gain = st.slider("gain (per frame)", 0.05, 1.0, 0.5, 0.05)
        delay = st.slider("delay (frames)", 0, 6, 2, 1)
        noise = st.slider("noise (rad per frame)", 0.0, 0.3, 0.05, 0.01)
        switch = st.checkbox("switch to pure pursuit at frame 60", value=False)
        seed = st.number_input("seed", 0, 9999, 1, 1)
    out = run_playground(target_kind, law, tau, N, gain, delay, noise, switch, seed)
    with c2:
        if out is None:
            st.warning("That chase ended at once; change the seed or the target.")
            return
        n = out["n_frames"]
        upto = st.slider("play", 2, n, n, 1)
        k = int(np.searchsorted(out["s"], upto - 1, side="right"))
        full_path = np.full(n, int(out["path"][0]))
        full_path[out["s"]] = out["path"]
        full_path[out["s"][-1] + 1 :] = out["path"][-1]
        st.pyplot(
            draw_chase(
                out["px"],
                out["py"],
                out["tx"],
                out["ty"],
                full_path,
                upto,
                "simulated chase",
                out["fit"]["roles"],
            )
        )
        probs = [out["gamma"][:, j] for j in range(out["gamma"].shape[1])]
        st.pyplot(draw_posterior(out["t"], probs, max(k, 2), out["law_true"]))
        f = out["fit"]
        st.write(
            f"Fitted on this chase: gain {f['gain']:.3f} per frame, lead time {f['tau']:.2f} s, "
            f"navigation constant {f['N']:.2f}, noise {f['sigma']:.3f} rad; states {', '.join(f['roles'])}."
        )


# ----------------------------------------------------------------------------- results browser


@st.cache_data(show_spinner=False)
def load_results(dataset: str):
    base = RESULTS / "fits" / dataset
    chase = pd.read_csv(base / "posteriors_chase.csv")
    frames = pd.read_csv(base / "posteriors_frames.csv")
    with open(base / "fit_report.json", encoding="utf-8") as f:
        rep = json.load(f)
    return chase, frames, rep


def page_browser():
    st.header("Results browser: real mouse chases")
    st.write(
        "These are real chases from three public recordings: two of mice hunting crickets and one of mice "
        "chasing other mice. Pick a chase and drag the play slider to replay it. The pursuer's path is "
        "coloured by the rule the model decoded, and the strip below shows how sure the model was at each "
        "moment. On the right are the values fitted to the whole dataset, with their uncertainty, and a "
        "short summary of the chase you picked."
    )
    datasets = [d for d in LABELS if (RESULTS / "fits" / d / "posteriors_frames.csv").exists()]
    if not datasets:
        st.info("No fitted results found under results/fits.")
        return
    dataset = st.selectbox("dataset", datasets, format_func=lambda d: LABELS[d])
    chase, frames, rep = load_results(dataset)
    available = set(frames["chase_id"].unique())
    chase = chase[chase["chase_id"].isin(available)]
    animals = sorted(chase["animal_id"].unique())
    animal = st.selectbox("animal", animals)
    sub = chase[chase["animal_id"] == animal].sort_values("n_frames", ascending=False)
    lengths = sub.set_index("chase_id")["n_frames"]
    cid = st.selectbox(
        "chase (longest first)",
        sub["chase_id"].tolist(),
        format_func=lambda c: f"{c} ({int(lengths.loc[c])} frames)",
    )
    g = frames[frames["chase_id"] == cid].sort_values("frame").reset_index(drop=True)
    roles = rep.get("state_roles") or rep.get("posteriors", {}).get("state_roles") or []
    c1, c2 = st.columns([1.2, 1])
    with c1:
        n = len(g)
        upto = st.slider("play", 2, n, n, 1)
        probs = [g[c].to_numpy() for c in ("p_pure", "p_lead", "p_pn", "p_persistence") if c in g.columns]
        st.pyplot(
            draw_chase(
                g["px"].to_numpy(),
                g["py"].to_numpy(),
                g["tx"].to_numpy(),
                g["ty"].to_numpy(),
                g["path"].to_numpy().astype(int),
                upto,
                f"{LABELS[dataset]}: {cid}",
                roles,
            )
        )
        st.pyplot(draw_posterior(g["t"].to_numpy() - g["t"].iloc[0], probs, upto))
    with c2:
        bs = rep.get("bootstrap", {})
        st.subheader("Fitted parameters (development set)")
        rows = []
        for name, label in (
            ("gain", "gain per frame"),
            ("tau", "lead time (s)"),
            ("N", "navigation constant"),
            ("sigma", "noise (rad)"),
        ):
            if name in bs:
                rows.append(
                    {
                        "parameter": label,
                        "estimate": round(bs[name]["estimate"], 3),
                        "low": round(bs[name]["lo"], 3),
                        "high": round(bs[name]["hi"], 3),
                    }
                )
        st.table(pd.DataFrame(rows))
        st.write(
            f"Delay {rep.get('delay_frames')} frames ({rep.get('delay_ms', 0):.0f} ms). States: {', '.join(roles) if roles else 'see the report'}."
        )
        row = sub.set_index("chase_id").loc[cid]
        if "p_steering_mean" in row:
            st.write(
                f"This chase: steering probability {row['p_steering_mean']:.2f} over {int(row['n_frames'])} frames "
                f"({int(row['n_informative'])} with the target moving), {int(row['n_switches'])} decoded switches, ended by {row['end_reason']}."
            )
        comp = {r["model"]: r for r in rep.get("comparison_held", [])}
        if "switching:plain" in comp:
            r = comp["switching:plain"]
            st.write(
                f"Dataset: the switching model beats the best single law by {r['diff_vs_best_single_per_frame']:+.4f} nats per frame "
                f"({r['diff_lo']:+.4f} to {r['diff_hi']:+.4f}) on {r['n_chases']} held-out chases."
            )


page = st.sidebar.radio("page", ["Playground", "Results browser"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"Developed by Reza Azad Gholami.\n\nCode, results and the paper: "
    f"[github.com/eigenreza/markov-switching-pursuit]({REPO})"
)
st.sidebar.caption(
    "The real chases come from public recordings: Michaiel, Abe and Niell (2020) and Holmgren et al. "
    "(2021), mice hunting crickets, and the MABe22 benchmark of Sun et al., mice chasing mice. Results "
    "derived from MABe22 are shared under "
    "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)."
)

st.title("Anticipate or react?")
st.write(
    "When a mouse chases its prey, does it run at where the prey is, or at where the prey is about to be? "
    "Pursuit theory offers a few steering rules for this. Pure pursuit aims straight at the target, lead "
    "pursuit aims at where the target will be a moment later, and proportional navigation turns just enough "
    "to keep the target at a steady bearing. This app lets you watch a model work out, moment by moment, "
    "which of these rules a pursuer is following, and browse what it found in real recordings of mice. "
    "Switch between the two pages on the left."
)
if page == "Playground":
    page_playground()
else:
    page_browser()
