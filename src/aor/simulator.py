"""Pursuer-target simulator.

The simulator produces chases in the same format as the real-data loaders, with the true
generating law at every frame kept alongside. The pursuer steers by one of the library laws,
or switches between them, through a fixed motor delay and a steering gain with Gaussian noise on
the heading increment; its speed follows a first-order autoregressive process. The target
follows one of several models, some of which the inference does not know about, so that
misspecification can be tested.

Target models
- constant_velocity: constant speed, heading changed at random turn times.
- evader: as constant_velocity, but turns away from the pursuer when the range falls below a
  threshold.
- jumper: mostly still or crawling, with sudden displacements at random times, directed away
  from the pursuer (cricket-like).

Switching rules
- none: a single law throughout.
- fixed: a list of (frame, law) switch points.
- markov: a transition matrix applied every frame.
- manoeuvre_triggered: a base law, with a switch to a reaction law for a fixed number of frames
  after every target manoeuvre.
- range_triggered: one law beyond a range threshold, another inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import Chase, ChaseMeta, build_frames, wrap_angle
from .kinematics import speed as speed_of
from .kinematics import velocity
from .laws import LAW_INDEX, LAWS, LawParams, command_step


@dataclass(frozen=True)
class TargetParams:
    kind: str = "constant_velocity"
    speed: float = 8.0  # cm/s
    turn_rate_hz: float = 0.5  # expected random turns per second (constant_velocity, evader)
    turn_sd: float = 1.0  # standard deviation of the heading change at a turn, radians
    evade_range: float = 12.0  # cm, evader turns away inside this range
    evade_gain: float = 0.6  # fraction of the away-error corrected per frame when evading
    jump_rate_hz: float = 0.6  # expected jumps per second (jumper)
    jump_length: float = 8.0  # cm per jump
    jump_frames: int = 3  # frames over which a jump is spread
    jump_spread_rad: float = 1.0  # half-width of the random jump direction around the away direction
    crawl_speed: float = 0.5  # cm/s between jumps (jumper)
    heading_noise: float = 0.05  # radians per frame on the target heading


@dataclass(frozen=True)
class PursuerParams:
    law: str = "pure_pursuit"
    params: LawParams = field(default_factory=LawParams)
    speed_mean: float = 20.0  # cm/s
    speed_ar: float = 0.9  # autoregressive coefficient of speed deviations
    speed_sd: float = 2.0  # innovation standard deviation of the speed process
    law_params: dict[str, LawParams] | None = None  # per-law overrides when switching
    vel_window: int = (
        2  # the pursuer perceives the target velocity as the backward difference over this many frames
    )


@dataclass(frozen=True)
class SwitchParams:
    kind: str = "none"
    fixed: tuple[tuple[int, str], ...] = ()
    transition: tuple[tuple[float, ...], ...] | None = None  # K by K row-stochastic
    base_law: str = "lead_pursuit"
    reaction_law: str = "pure_pursuit"
    reaction_frames: int = 10
    range_threshold: float = 15.0
    far_law: str = "proportional_navigation"
    near_law: str = "pure_pursuit"


@dataclass(frozen=True)
class WorldParams:
    fps: float = 30.0
    arena: tuple[float, float] = (45.0, 38.0)  # cm
    capture_radius: float = 2.0  # cm
    max_frames: int = 300
    obs_pos_sd: float = 0.0  # cm, measurement noise on exported positions
    obs_heading_sd: float = 0.0  # radians, measurement noise on exported heading
    obs_glitch_rate: float = 0.0  # per frame, probability of a single-frame heading glitch in the export
    obs_glitch_sd: float = 1.0  # radians, size of a glitch


@dataclass
class SimResult:
    """Everything the simulator knows: true states, true law per frame, manoeuvre flags."""

    fps: float
    px: np.ndarray
    py: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    t_heading: np.ndarray
    t_speed: np.ndarray
    law_index: np.ndarray
    manoeuvre: np.ndarray
    command: np.ndarray  # command applied at each frame (after the delay)
    end_reason: str
    target: TargetParams
    pursuer: PursuerParams
    switching: SwitchParams
    world: WorldParams
    seed: int

    @property
    def n_frames(self) -> int:
        return len(self.px)


def _reflect(x: float, lo: float, hi: float) -> tuple[float, bool]:
    if x < lo:
        return 2 * lo - x, True
    if x > hi:
        return 2 * hi - x, True
    return x, False


class _Target:
    def __init__(
        self,
        p: TargetParams,
        world: WorldParams,
        rng: np.random.Generator,
        x: float,
        y: float,
        heading: float,
    ):
        self.p, self.world, self.rng = p, world, rng
        self.x, self.y, self.heading = x, y, heading
        self.speed = p.speed if p.kind != "jumper" else p.crawl_speed
        self.jump_left = 0
        self.jump_dir = 0.0
        self.vx = self.speed * np.cos(heading)
        self.vy = self.speed * np.sin(heading)
        self.history = [(x, y)]

    def perceived_velocity(self, window: int) -> tuple[float, float]:
        """Backward-difference velocity over ``window`` frames, from the position history."""
        w = max(1, int(window))
        if len(self.history) <= w:
            return self.vx, self.vy
        (x0, y0), (x1, y1) = self.history[-1 - w], self.history[-1]
        dt = w / self.world.fps
        return (x1 - x0) / dt, (y1 - y0) / dt

    def step(self, px: float, py: float) -> bool:
        """Advance one frame; return True if this frame holds a manoeuvre."""
        p, dt = self.p, 1.0 / self.world.fps
        manoeuvre = False
        away = float(wrap_angle(np.arctan2(self.y - py, self.x - px)))
        rng = np.hypot(self.x - px, self.y - py)
        if p.kind in ("constant_velocity", "evader"):
            if self.rng.random() < p.turn_rate_hz * dt:
                self.heading = float(wrap_angle(self.heading + self.rng.normal(0.0, p.turn_sd)))
                manoeuvre = True
            if p.kind == "evader" and rng < p.evade_range:
                err = float(wrap_angle(away - self.heading))
                if abs(err) > 0.3:
                    manoeuvre = manoeuvre or abs(p.evade_gain * err) > 0.3
                self.heading = float(wrap_angle(self.heading + p.evade_gain * err))
            self.heading = float(wrap_angle(self.heading + self.rng.normal(0.0, p.heading_noise)))
            self.speed = p.speed
        elif p.kind == "jumper":
            if self.jump_left > 0:
                self.jump_left -= 1
                self.speed = p.jump_length / (p.jump_frames * dt)
                self.heading = self.jump_dir
            else:
                if self.rng.random() < p.jump_rate_hz * dt:
                    self.jump_dir = float(
                        wrap_angle(away + self.rng.uniform(-p.jump_spread_rad, p.jump_spread_rad))
                    )
                    self.jump_left = p.jump_frames - 1
                    self.speed = p.jump_length / (p.jump_frames * dt)
                    self.heading = self.jump_dir
                    manoeuvre = True
                else:
                    self.speed = p.crawl_speed
                    self.heading = float(wrap_angle(self.heading + self.rng.normal(0.0, 0.3)))
        else:
            raise ValueError(f"unknown target kind {p.kind!r}")
        nx = self.x + self.speed * np.cos(self.heading) * dt
        ny = self.y + self.speed * np.sin(self.heading) * dt
        nx, bx = _reflect(nx, 0.0, self.world.arena[0])
        ny, by = _reflect(ny, 0.0, self.world.arena[1])
        if bx:
            self.heading = float(wrap_angle(np.pi - self.heading))
            self.jump_dir = self.heading
        if by:
            self.heading = float(wrap_angle(-self.heading))
            self.jump_dir = self.heading
        self.vx, self.vy = (nx - self.x) / dt, (ny - self.y) / dt
        self.x, self.y = nx, ny
        self.history.append((nx, ny))
        return manoeuvre


class _Switcher:
    def __init__(self, s: SwitchParams, start_law: str, rng: np.random.Generator):
        self.s, self.rng = s, rng
        self.law = start_law
        self.reaction_left = 0
        if s.kind == "fixed":
            self.fixed = sorted(s.fixed)
        if s.kind == "manoeuvre_triggered":
            self.law = s.base_law
        if s.kind == "markov" and s.transition is None:
            raise ValueError("markov switching needs a transition matrix")

    def step(self, frame: int, rng_cm: float, manoeuvre: bool) -> str:
        s = self.s
        if s.kind == "none":
            return self.law
        if s.kind == "fixed":
            for f, law in self.fixed:
                if frame >= f:
                    self.law = law
            return self.law
        if s.kind == "markov":
            row = np.asarray(s.transition)[LAW_INDEX[self.law]]
            self.law = LAWS[self.rng.choice(len(LAWS), p=row / row.sum())]
            return self.law
        if s.kind == "manoeuvre_triggered":
            if manoeuvre:
                self.reaction_left = s.reaction_frames
            if self.reaction_left > 0:
                self.reaction_left -= 1
                self.law = s.reaction_law
            else:
                self.law = s.base_law
            return self.law
        if s.kind == "range_triggered":
            self.law = s.far_law if rng_cm > s.range_threshold else s.near_law
            return self.law
        raise ValueError(f"unknown switching kind {s.kind!r}")


def simulate(
    target: TargetParams = TargetParams(),
    pursuer: PursuerParams = PursuerParams(),
    switching: SwitchParams = SwitchParams(),
    world: WorldParams = WorldParams(),
    seed: int = 0,
    start: tuple[float, float, float, float, float, float] | None = None,
) -> SimResult:
    """Run one chase. ``start`` is (px, py, heading, tx, ty, target heading); when omitted the
    pursuer starts near one wall facing the arena centre and the target at a random position
    at least 10 cm away."""
    rng = np.random.default_rng(seed)
    W, H = world.arena
    if start is None:
        px, py = rng.uniform(0.1 * W, 0.3 * W), rng.uniform(0.2 * H, 0.8 * H)
        while True:
            tx, ty = rng.uniform(0.4 * W, 0.9 * W), rng.uniform(0.1 * H, 0.9 * H)
            if np.hypot(tx - px, ty - py) >= 10.0:
                break
        heading = float(wrap_angle(np.arctan2(ty - py, tx - px) + rng.normal(0.0, 0.5)))
        t_heading = rng.uniform(-np.pi, np.pi)
    else:
        px, py, heading, tx, ty, t_heading = start
    tgt = _Target(target, world, rng, tx, ty, t_heading)
    sw = _Switcher(switching, pursuer.law, rng)
    law_params = dict(pursuer.law_params or {})
    dt = 1.0 / world.fps
    speed = pursuer.speed_mean
    n = world.max_frames
    out = {
        k: np.zeros(n)
        for k in ("px", "py", "heading", "speed", "tx", "ty", "t_heading", "t_speed", "command")
    }
    law_index = np.zeros(n, dtype=int)
    manoeuvre = np.zeros(n, dtype=bool)
    buffer: list[float] = []
    prev_los = None
    end_reason = "clip_end"
    n_done = n
    noise_prev = 0.0
    for i in range(n):
        rng_cm = float(np.hypot(tgt.x - px, tgt.y - py))
        man = bool(manoeuvre[i - 1]) if i > 0 else False
        law = sw.step(i, rng_cm, man)
        p = law_params.get(law, pursuer.params)
        out["px"][i], out["py"][i], out["heading"][i], out["speed"][i] = px, py, heading, speed
        out["tx"][i], out["ty"][i], out["t_heading"][i], out["t_speed"][i] = (
            tgt.x,
            tgt.y,
            tgt.heading,
            tgt.speed,
        )
        law_index[i] = LAW_INDEX[law]
        if rng_cm <= world.capture_radius:
            end_reason = "capture"
            n_done = i + 1
            break
        pvx, pvy = tgt.perceived_velocity(pursuer.vel_window)
        cmd, prev_los = command_step(law, px, py, heading, tgt.x, tgt.y, pvx, pvy, p, prev_los)
        buffer.append(cmd)
        applied = buffer[-1 - p.delay] if len(buffer) > p.delay else 0.0
        out["command"][i] = applied
        noise_now = p.noise_ar * noise_prev + rng.normal(0.0, p.noise)
        noise_prev = noise_now
        heading = float(wrap_angle(heading + applied + noise_now))
        speed = max(
            0.0,
            pursuer.speed_mean
            + pursuer.speed_ar * (speed - pursuer.speed_mean)
            + rng.normal(0.0, pursuer.speed_sd),
        )
        px = float(np.clip(px + speed * np.cos(heading) * dt, 0.0, W))
        py = float(np.clip(py + speed * np.sin(heading) * dt, 0.0, H))
        manoeuvre[i] = tgt.step(px, py)
    sl = slice(0, n_done)
    return SimResult(
        world.fps,
        out["px"][sl],
        out["py"][sl],
        out["heading"][sl],
        out["speed"][sl],
        out["tx"][sl],
        out["ty"][sl],
        out["t_heading"][sl],
        out["t_speed"][sl],
        law_index[sl],
        manoeuvre[sl],
        out["command"][sl],
        end_reason,
        target,
        pursuer,
        switching,
        world,
        seed,
    )


def simulate_along_target(
    tx: np.ndarray,
    ty: np.ndarray,
    fps: float,
    start_px: float,
    start_py: float,
    start_heading: float,
    speeds: np.ndarray,
    law: str,
    params: LawParams,
    seed: int = 0,
    capture_radius: float | None = None,
    vel_window: int = 2,
    obs_pos_sd: float = 0.0,
    obs_heading_sd: float = 0.0,
    collision_course: bool = False,
) -> dict:
    """Run a pursuer along a recorded target path. The pursuer starts where the recorded pursuer
    started and moves at the recorded speed profile; its heading follows ``law`` through the delay
    and noise in ``params``, or, with ``collision_course``, turns straight onto the constant-bearing
    course (the optimal pursuer, no delay). Stops at capture when a radius is given. Returns the
    pursuer path, headings, the frame of capture (or None), and optionally noisy observations."""
    from .laws import collision_course_heading

    rng = np.random.default_rng(seed)
    n = len(tx)
    dt = 1.0 / fps
    px, py, heading = np.zeros(n), np.zeros(n), np.zeros(n)
    px[0], py[0], heading[0] = start_px, start_py, start_heading
    buffer: list[float] = []
    prev_los = None
    capture = None
    w = max(1, vel_window)
    for i in range(n - 1):
        if (
            capture is None
            and capture_radius is not None
            and np.hypot(tx[i] - px[i], ty[i] - py[i]) <= capture_radius
        ):
            capture = i
        j = max(0, i - w)
        tvx = (tx[i] - tx[j]) / ((i - j) * dt) if i > j else 0.0
        tvy = (ty[i] - ty[j]) / ((i - j) * dt) if i > j else 0.0
        if collision_course:
            heading[i + 1] = float(
                collision_course_heading(px[i], py[i], tx[i], ty[i], tvx, tvy, max(speeds[i], 1e-6))
            )
        else:
            cmd, prev_los = command_step(
                law, px[i], py[i], heading[i], tx[i], ty[i], tvx, tvy, params, prev_los
            )
            buffer.append(cmd)
            applied = buffer[-1 - params.delay] if len(buffer) > params.delay else 0.0
            heading[i + 1] = float(wrap_angle(heading[i] + applied + rng.normal(0.0, params.noise)))
        px[i + 1] = px[i] + speeds[i] * np.cos(heading[i + 1]) * dt
        py[i + 1] = py[i] + speeds[i] * np.sin(heading[i + 1]) * dt
    if (
        capture is None
        and capture_radius is not None
        and np.hypot(tx[-1] - px[-1], ty[-1] - py[-1]) <= capture_radius
    ):
        capture = n - 1
    obs = {}
    if obs_pos_sd > 0 or obs_heading_sd > 0:
        obs = {
            "px": px + rng.normal(0, obs_pos_sd, n),
            "py": py + rng.normal(0, obs_pos_sd, n),
            "tx": tx + rng.normal(0, obs_pos_sd, n),
            "ty": ty + rng.normal(0, obs_pos_sd, n),
            "heading": wrap_angle(heading + rng.normal(0, obs_heading_sd, n)),
        }
    return {"px": px, "py": py, "heading": heading, "capture_frame": capture, "obs": obs}


def to_chase(
    res: SimResult,
    chase_id: str,
    session_id: str = "synthetic",
    animal_id: str = "sim",
    seed: int | None = None,
) -> Chase:
    """Export a simulation in the real-data format: observed positions (with optional measurement
    noise), heading, smoothed speed, and target velocity by the same finite difference the real
    pipeline uses. The true law index travels in the optional column ``law_true``."""
    rng = np.random.default_rng(res.seed if seed is None else seed)
    n = res.n_frames
    w = res.world
    px = res.px + rng.normal(0.0, w.obs_pos_sd, n) if w.obs_pos_sd > 0 else res.px.copy()
    py = res.py + rng.normal(0.0, w.obs_pos_sd, n) if w.obs_pos_sd > 0 else res.py.copy()
    tx = res.tx + rng.normal(0.0, w.obs_pos_sd, n) if w.obs_pos_sd > 0 else res.tx.copy()
    ty = res.ty + rng.normal(0.0, w.obs_pos_sd, n) if w.obs_pos_sd > 0 else res.ty.copy()
    heading = (
        wrap_angle(res.heading + rng.normal(0.0, w.obs_heading_sd, n))
        if w.obs_heading_sd > 0
        else res.heading.copy()
    )
    if w.obs_glitch_rate > 0:
        hit = rng.random(n) < w.obs_glitch_rate
        heading = wrap_angle(heading + hit * rng.normal(0.0, w.obs_glitch_sd, n))
    pxy = np.column_stack([px, py])
    txy = np.column_stack([tx, ty])
    p_speed = speed_of(pxy, res.fps, window=5)
    t_vel = velocity(txy, res.fps, window=5)
    frames = np.arange(n)
    df = build_frames(
        frames,
        res.fps,
        px,
        py,
        heading,
        p_speed,
        tx,
        ty,
        t_vel[:, 0],
        t_vel[:, 1],
        t_speed=np.hypot(t_vel[:, 0], t_vel[:, 1]),
        t_heading=res.t_heading,
        law_true=res.law_index.astype(float),
    )
    meta = ChaseMeta(
        "synthetic",
        session_id,
        animal_id,
        chase_id,
        res.fps,
        0,
        n - 1,
        res.end_reason,
        "simulated",
        pursuer_point="simulated_point",
        px_per_cm=None,
        native_fps=res.fps,
        coordinate_frame="arena_cm",
        extra={
            "seed": res.seed,
            "target": res.target.__dict__,
            "switching": res.switching.kind,
            "law": res.pursuer.law,
            "law_params": res.pursuer.params.__dict__,
        },
    )
    return Chase(meta=meta, frames=df)
