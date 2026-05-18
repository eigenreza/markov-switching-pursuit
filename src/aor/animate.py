"""Short animations for the README, drawn from committed result files with matplotlib and
written as GIF: a real chase with the decoded state colouring the pursuer's track, a simulated
chase with a scripted switch and the posterior catching it, and the manoeuvre-aligned curve
building up."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .figures.style import GREY, LAW_COLOURS, apply_style
from .laws import LAWS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "animations"
STATE_COLOURS = [LAW_COLOURS[law] for law in LAWS] + ["#999999"]
STATE_NAMES = ["pure pursuit", "lead pursuit", "proportional navigation", "persistence"]
FPS_GIF = 15


def _longest_chase(dataset: str = "michaiel2020", rank: int = 0) -> pd.DataFrame | None:
    path = RESULTS / "fits" / dataset / "posteriors_frames.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    counts = f.groupby("chase_id").size().sort_values(ascending=False)
    if len(counts) <= rank:
        return None
    return f[f["chase_id"] == counts.index[rank]].sort_values("frame").reset_index(drop=True)


def animate_real_chase(
    out_dir: Path = OUT, dataset: str = "michaiel2020", rank: int = 1, stride: int = 1
) -> Path | None:
    """A held-out chase played frame by frame: the pursuer's track coloured by the decoded state,
    the target's track in black, and the posterior filling in underneath."""
    g = _longest_chase(dataset, rank)
    if g is None:
        return None
    apply_style()
    n = len(g)
    frames = list(range(2, n + 1, stride))
    px, py, tx, ty = (g[c].to_numpy() for c in ("px", "py", "tx", "ty"))
    t = g["t"].to_numpy()
    t = t - t[0]
    probs = [g[c].to_numpy() for c in ("p_pure", "p_lead", "p_pn") if c in g.columns]
    path = g["path"].to_numpy().astype(int)
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(4.2, 5.2), gridspec_kw={"height_ratios": [3, 1.2]})
    pad = 2.0
    ax.set_xlim(min(px.min(), tx.min()) - pad, max(px.max(), tx.max()) + pad)
    ax.set_ylim(min(py.min(), ty.min()) - pad, max(py.max(), ty.max()) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("mouse and cricket, held-out chase", fontsize=9)
    (target_line,) = ax.plot([], [], "-", color="#000000", lw=1.2)
    (target_dot,) = ax.plot([], [], "o", color="#000000", ms=5)
    (pursuer_dot,) = ax.plot([], [], "s", color=GREY, ms=6)
    segs = []
    axp.set_xlim(0, t[-1])
    axp.set_ylim(0, 1)
    axp.set_xlabel("time (s)")
    axp.set_ylabel("posterior")
    fills: list = []
    roles = []
    rep_path = RESULTS / "fits" / dataset / "fit_report.json"
    if rep_path.exists():
        with open(rep_path, encoding="utf-8") as f:
            roles = json.load(f).get("state_roles", [])
    for k, name in enumerate(STATE_NAMES[: len(probs)]):
        label = f"{name} ({roles[k]})" if k < len(roles) else name
        axp.plot([], [], color=STATE_COLOURS[k], lw=4, label=label)
    axp.legend(frameon=False, fontsize=6, loc="upper right", ncol=1)

    def draw(i):
        for s_ in segs:
            s_.remove()
        segs.clear()
        for f_ in fills:
            f_.remove()
        fills.clear()
        target_line.set_data(tx[:i], ty[:i])
        target_dot.set_data([tx[i - 1]], [ty[i - 1]])
        pursuer_dot.set_data([px[i - 1]], [py[i - 1]])
        # pursuer track in runs of constant decoded state
        start = 0
        for j in range(1, i):
            if path[j] != path[start] or j == i - 1:
                end = j if path[j] != path[start] else j + 1
                (ln,) = ax.plot(px[start:end], py[start:end], "-", color=STATE_COLOURS[path[start]], lw=2.2)
                segs.append(ln)
                start = j
        bottom = np.zeros(i)
        for k, pr in enumerate(probs):
            fl = axp.fill_between(t[:i], bottom, bottom + pr[:i], color=STATE_COLOURS[k], alpha=0.85, lw=0)
            fills.append(fl)
            bottom = bottom + pr[:i]
        return [target_line, target_dot, pursuer_dot, *segs, *fills]

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "real_chase.gif"
    anim.save(out, writer=PillowWriter(fps=FPS_GIF), dpi=110)
    plt.close(fig)
    return out


def simulated_switch_example(seed: int = 3) -> dict:
    """One simulated chase with a scripted switch from lead pursuit to pure pursuit, decoded by a
    switching model fitted to a few chases of the same kind."""
    from dataclasses import replace

    from .inference.models import FitSettings, fit_switching, posteriors
    from .laws import LawParams
    from .simulator import PursuerParams, SwitchParams, TargetParams, WorldParams, simulate, to_chase

    params = LawParams(gain=0.5, delay=2, noise=0.05, tau=0.5, N=3.0)
    sw = SwitchParams(kind="fixed", fixed=((60, "pure_pursuit"),), base_law="lead_pursuit")
    chases = []
    for i in range(12):
        res = simulate(
            TargetParams(kind="constant_velocity", speed=10.0, turn_rate_hz=0.6),
            PursuerParams(law="lead_pursuit", params=params, speed_mean=16.0),
            sw,
            WorldParams(arena=(80.0, 80.0), max_frames=150, capture_radius=1.5),
            seed=seed + i,
        )
        c = to_chase(res, f"anim{i}", animal_id="sim")
        if len(c.frames) >= 90:
            chases.append(c)
    fit = fit_switching(chases, FitSettings(delay=2, n_iter=40, n_restarts=1))
    post = posteriors(chases[:1], fit)[0]
    f = chases[0].frames
    return {
        "px": f["px"].to_numpy(),
        "py": f["py"].to_numpy(),
        "tx": f["tx"].to_numpy(),
        "ty": f["ty"].to_numpy(),
        "s": post["s"],
        "gamma": post["gamma"],
        "law_true": post["law_true"],
        "fps": chases[0].meta.fps,
        "fit": replace(fit),
    }


def animate_simulated_switch(out_dir: Path = OUT, stride: int = 1) -> Path | None:
    """A simulated chase with a scripted switch from lead pursuit to pure pursuit, with the
    posterior catching the switch."""
    ex = simulated_switch_example()
    apply_style()
    px, py, tx, ty = ex["px"], ex["py"], ex["tx"], ex["ty"]
    gamma = np.asarray(ex["gamma"], dtype=float)
    s = np.asarray(ex["s"], dtype=int)
    truth = None if ex["law_true"] is None else np.asarray(ex["law_true"], dtype=int)
    fps = float(ex["fps"])
    n = len(s)
    frames = list(range(2, n + 1, stride))
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(4.2, 5.2), gridspec_kw={"height_ratios": [3, 1.2]})
    pad = 2.0
    ax.set_xlim(min(px.min(), tx.min()) - pad, max(px.max(), tx.max()) + pad)
    ax.set_ylim(min(py.min(), ty.min()) - pad, max(py.max(), ty.max()) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("simulated chase: lead pursuit, then pure pursuit", fontsize=9)
    (target_line,) = ax.plot([], [], "-", color="#000000", lw=1.2)
    (pursuer_line,) = ax.plot([], [], "-", color=GREY, lw=2)
    (pursuer_dot,) = ax.plot([], [], "s", color=GREY, ms=6)
    t = s / fps
    axp.set_xlim(t[0], t[-1])
    axp.set_ylim(0, 1)
    axp.set_xlabel("time (s)")
    axp.set_ylabel("posterior")
    fills: list = []
    (truth_line,) = axp.plot([], [], "k--", lw=0.8, label="true law (1 = pure)")
    for k in range(gamma.shape[1]):
        axp.plot([], [], color=STATE_COLOURS[k], lw=4, label=STATE_NAMES[k])
    axp.legend(frameon=False, fontsize=6, loc="upper right")

    def draw(i):
        for f_ in fills:
            f_.remove()
        fills.clear()
        j = s[i - 1] + 1
        target_line.set_data(tx[:j], ty[:j])
        pursuer_line.set_data(px[:j], py[:j])
        pursuer_dot.set_data([px[j - 1]], [py[j - 1]])
        bottom = np.zeros(i)
        for k in range(gamma.shape[1]):
            fl = axp.fill_between(
                t[:i], bottom, bottom + gamma[:i, k], color=STATE_COLOURS[k], alpha=0.85, lw=0
            )
            fills.append(fl)
            bottom = bottom + gamma[:i, k]
        if truth is not None:
            truth_line.set_data(t[:i], (truth[:i] == 0).astype(float) * 0.9 + 0.05)
        return [target_line, pursuer_line, pursuer_dot, *fills, truth_line]

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "simulated_switch.gif"
    anim.save(out, writer=PillowWriter(fps=FPS_GIF), dpi=110)
    plt.close(fig)
    return out


def animate_event_aligned(out_dir: Path = OUT, dataset: str = "mabe22") -> Path | None:
    """The manoeuvre-aligned steering probability building up lag by lag."""
    path = RESULTS / "fits" / dataset / "event_aligned.csv"
    if not path.exists():
        return None
    ev = pd.read_csv(path)
    if ev.empty or ev["mean"].isna().all():
        return None
    rep_path = RESULTS / "fits" / dataset / "fit_report.json"
    fps = 30.0
    if rep_path.exists():
        with open(rep_path, encoding="utf-8") as f:
            fps = float(json.load(f)["dev"]["fps"])
    apply_style()
    t = ev["lag_frames"].to_numpy() / fps
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("time from target manoeuvre (s)")
    ax.set_ylabel("steering probability")
    ax.set_title("mouse and mouse: around a target manoeuvre", fontsize=9)
    (line,) = ax.plot([], [], "-", color=LAW_COLOURS["pure_pursuit"], lw=1.6)
    band: list = []

    def draw(i):
        for b in band:
            b.remove()
        band.clear()
        line.set_data(t[:i], ev["mean"].to_numpy()[:i])
        fl = ax.fill_between(
            t[:i],
            ev["lo"].to_numpy()[:i],
            ev["hi"].to_numpy()[:i],
            color=LAW_COLOURS["pure_pursuit"],
            alpha=0.25,
            lw=0,
        )
        band.append(fl)
        return [line, fl]

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=range(2, len(t) + 1), blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "event_aligned.gif"
    anim.save(out, writer=PillowWriter(fps=FPS_GIF), dpi=110)
    plt.close(fig)
    return out


def strip_gif_metadata(path: Path) -> None:
    """Rewrite a GIF without comment or application blocks that could carry metadata."""
    from PIL import Image, ImageSequence

    im = Image.open(path)
    frames = [fr.copy() for fr in ImageSequence.Iterator(im)]
    duration = im.info.get("duration", int(1000 / FPS_GIF))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2)


def make_all(out_dir: Path = OUT) -> list[Path]:
    out = []
    for fn in (animate_real_chase, animate_simulated_switch, animate_event_aligned):
        p = fn(out_dir)
        if p is not None:
            strip_gif_metadata(p)
            out.append(p)
    return out
