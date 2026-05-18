"""Project task runner.

Usage: python tasks.py <command> [options]

Commands
  setup      install the package in editable mode with development extras
  data       download and verify raw datasets (use --verify to check without downloading)
  audit      run the dataset audits and write results/audit/<dataset>.json
  simulate   generate the synthetic suite under data/derived/synthetic/
  recover    run the recovery study on the synthetic suite, writing results/recovery/
  figures    draw every figure whose inputs exist into results/figures/
  fit        fit the models on the real data (development set, held-out pass, replication), writing results/fits/
  evaluate   optimal-play benchmark and two-player test on the held-out chases, writing results/evaluate/
  tables     write the result tables under results/tables/ from the committed result files
  test       run the test suite
  lint       run ruff on the package and tests
  check      style and consistency checks on public files
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

PUBLIC_SUFFIXES = {
    ".py",
    ".md",
    ".tex",
    ".bib",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".cfg",
    ".ipynb",
    ".csv",
}
# The pattern lists are assembled from fragments so that this file does not itself contain the strings it hunts.
FORBIDDEN_STRINGS = [
    "co-" + "authored",
    "gener" + "ated by",
    "assis" + "tant",
    "ag" + "ent",
    "anticipation " + "probability",
]
WARN_WORDS = [
    "del" + "ve",
    "lever" + "age",
    "rob" + "ust",
    "novel fra" + "mework",
    "para" + "digm",
    "cutting" + "-edge",
    "state-of" + "-the-art",
]
DATE_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),
    re.compile(
        r"\b(0?[1-9]|[12]\d|3[01])\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20)\d{2}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12]\d|3[01]),?\s+(19|20)\d{2}\b",
        re.IGNORECASE,
    ),
]


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def python() -> str:
    return sys.executable


def cmd_setup(_: argparse.Namespace) -> int:
    return run([python(), "-m", "pip", "install", "-e", ".[dev]"])


def cmd_data(args: argparse.Namespace) -> int:
    from aor.data.download import download_dataset

    rc = 0
    for name in args.datasets:
        for line in download_dataset(name, verify_only=args.verify):
            print(line)
            if line.startswith(("BAD", "missing")):
                rc = 1
    return rc


def cmd_audit(args: argparse.Namespace) -> int:
    from aor.data import RAW_DIR, holmgren, mabe22, michaiel

    modules = {"michaiel2020": michaiel, "holmgren2021": holmgren, "mabe22": mabe22}
    out_dir = ROOT / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in args.datasets:
        raw = RAW_DIR / name
        if not raw.exists():
            print(f"{name}: raw data not found at {raw}, run `python tasks.py data {name}` first")
            rc = 1
            continue
        report = modules[name].audit(raw)
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"{name}: wrote {path.relative_to(ROOT)}")
    return rc


def cmd_simulate(_: argparse.Namespace) -> int:
    from aor.synthetic import OUT_DIR, generate

    summary = generate()
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT_DIR.relative_to(ROOT)}")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    from aor.recovery import OUT_DIR, run

    summary = run(max_conditions=args.max_conditions)
    print(json.dumps(summary.get("limits", {}), indent=1))
    print(f"wrote {OUT_DIR.relative_to(ROOT)}")
    return 0


def cmd_figures(_: argparse.Namespace) -> int:
    from aor.figures import make_all

    report = make_all()
    for name, paths in report["written"].items():
        print("wrote", name, *paths)
    for name, missing in report["skipped"].items():
        print("skipped", name, "(missing inputs:", *missing, ")")
    return 0


FIT_PLAN = {"michaiel2020": ("dev", "held"), "holmgren2021": ("replication", None), "mabe22": ("dev", "held")}


def cmd_fit(args: argparse.Namespace) -> int:
    from aor.analysis.fit_real import RESULTS, run_dataset

    for name in args.datasets:
        dev, held = FIT_PLAN[name]
        run_dataset(name, dev, held, null_B=args.null_b, jobs=args.jobs, reuse_null=args.reuse_null)
    print(f"wrote {RESULTS.relative_to(ROOT)}")
    return 0


def _pick(block, name: str, default: int) -> int:
    if isinstance(block, dict):
        return int(block.get(name, block.get("default", default)))
    return int(default if block is None else block)


def cmd_followup(args: argparse.Namespace) -> int:
    """The post-freeze follow-up: the library with a persistence state, same splits, same steps."""
    from aor.analysis.fit_real import RESULTS_FOLLOWUP, keep_report_only, load_cfg, run_dataset

    fu = load_cfg().get("followup", {})
    for name in args.datasets:
        dev, held = FIT_PLAN[name]
        run_dataset(
            name,
            dev,
            held,
            persistence=True,
            null_B=_pick(fu.get("null_B"), name, 30),
            bootstrap_B=_pick(fu.get("bootstrap_params"), name, 50),
            bins_B=_pick(fu.get("bootstrap_bins"), name, 15),
            reuse_null=args.reuse_null,
        )
        keep_report_only(RESULTS_FOLLOWUP / name)
    print(f"wrote {RESULTS_FOLLOWUP.relative_to(ROOT)}")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Sensitivity analyses: the arena-fit scale of the primary dataset at the frozen delay, and
    the leave-one-animal-out score of the replication set."""
    import json

    from aor.analysis.fit_real import RESULTS, keep_report_only, leave_one_animal_out, load_cfg, run_dataset

    se = load_cfg().get("sensitivity", {})
    which = set(args.which) if args.which else {"scale", "loao"}
    unknown = which - {"scale", "loao"}
    if unknown:
        raise SystemExit(f"unknown sensitivity analysis {sorted(unknown)}; choose from scale, loao")
    if "scale" in which:
        with open(RESULTS / "michaiel2020" / "fit_report.json", encoding="utf-8") as f:
            delay = int(json.load(f)["delay_frames"])
        run_dataset(
            "michaiel2020",
            "dev",
            "held",
            variant="arena_scale",
            frozen_delay=delay,
            null_B=_pick(se.get("null_B"), "michaiel2020", 30),
            bootstrap_B=_pick(se.get("bootstrap_params"), "michaiel2020", 50),
            bins_B=_pick(se.get("bootstrap_bins"), "michaiel2020", 15),
        )
        keep_report_only(RESULTS / "michaiel2020_arena_scale")
    if "loao" in which:
        with open(RESULTS / "holmgren2021" / "fit_report.json", encoding="utf-8") as f:
            delay = int(json.load(f)["delay_frames"])
        leave_one_animal_out("holmgren2021", "replication", delay)
    print(f"wrote {RESULTS.relative_to(ROOT)}")
    return 0


def cmd_posteriors(args: argparse.Namespace) -> int:
    """Recompute the posterior tables and the event-aligned curve from the stored fits."""
    from aor.analysis.fit_real import RESULTS_FOLLOWUP, keep_report_only, refresh_posteriors

    for name in args.datasets:
        dev, held = FIT_PLAN[name]
        refresh_posteriors(name, dev, held, persistence=args.followup)
        if args.followup:
            keep_report_only(RESULTS_FOLLOWUP / name)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from aor.analysis.evaluate import RESULTS, run_dataset

    for name in args.datasets:
        run_dataset(name)
    print(f"wrote {RESULTS.relative_to(ROOT)}")
    return 0


def cmd_tables(_: argparse.Namespace) -> int:
    from aor.tables import OUT, build

    counts = build()
    print(json.dumps(counts))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


def cmd_animate(_: argparse.Namespace) -> int:
    """Write the README animations (GIF) from the committed result files."""
    from aor.animate import OUT, make_all

    out = make_all()
    print(f"wrote {OUT.relative_to(ROOT)}: " + ", ".join(p.name for p in out))
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    """Run the Streamlit demo locally (its own requirements are in demo/requirements.txt)."""
    return run(
        [python(), "-m", "streamlit", "run", str(ROOT / "demo" / "app.py"), "--server.headless", "true"]
    )


def cmd_paper(args: argparse.Namespace) -> int:
    """Write paper/numbers.tex and paper/tables/ from the result files and build the PDF."""
    from aor.paper import PAPER, build

    out = build(pdf=not args.no_pdf)
    print(f"wrote {PAPER.relative_to(ROOT)}" + (f" and {out.name}" if out else ""))
    return 0


def cmd_test(_: argparse.Namespace) -> int:
    return run([python(), "-m", "pytest", "-q"])


def cmd_lint(_: argparse.Namespace) -> int:
    return run([python(), "-m", "ruff", "check", "src", "tests", "tasks.py"])


def _ignore_rules() -> list[str]:
    """Ignore patterns from .gitignore and, when present, from git's private exclude list."""
    rules = []
    for path in (ROOT / ".gitignore", ROOT / ".git" / "info" / "exclude"):
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                rule = raw.strip()
                if rule and not rule.startswith(("#", "!")):
                    rules.append(rule)
    return rules


def _ignored(rel: str, is_dir: bool, rules: list[str]) -> bool:
    """Whether a path relative to the repository root matches an ignore rule: a rule ending in a
    slash matches directories only; a rule containing a slash is anchored at the root, any other
    rule matches the file or directory name anywhere."""
    name = rel.rsplit("/", 1)[-1]
    for rule in rules:
        if rule.endswith("/") and not is_dir:
            continue
        pat = rule.strip("/")
        if ("/" in pat and fnmatch.fnmatchcase(rel, pat)) or (
            "/" not in pat and fnmatch.fnmatchcase(name, pat)
        ):
            return True
    return False


def public_files(rules: list[str] | None = None) -> list[Path]:
    """The files a push publishes (everything the ignore lists leave in), plus the LaTeX source of
    the published paper, whose wording the same style rules govern."""
    rules = _ignore_rules() if rules is None else rules
    files: list[Path] = []

    def walk(d: Path) -> None:
        for p in sorted(d.iterdir()):
            rel = p.relative_to(ROOT).as_posix()
            if p.name == ".git" or _ignored(rel, p.is_dir(), rules):
                continue
            if p.is_dir():
                walk(p)
            elif p.suffix.lower() in PUBLIC_SUFFIXES:
                files.append(p)

    walk(ROOT)
    for p in sorted((ROOT / "paper").rglob("*")):
        if p.is_file() and p.suffix.lower() in (".tex", ".bib") and p not in files:
            files.append(p)
    return files


def cmd_check(_: argparse.Namespace) -> int:
    failures, warnings = [], []
    files = public_files()
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = p.relative_to(ROOT)
        if chr(0x2014) in text:
            failures.append(f"{rel}: em dash")
        is_bib = p.suffix == ".bib"
        if not is_bib:
            for pat in DATE_PATTERNS:
                m = pat.search(text)
                if m:
                    failures.append(f"{rel}: date pattern {m.group(0)!r}")
                    break
        low = text.lower()
        for s in FORBIDDEN_STRINGS:
            for m in re.finditer(r"\b" + re.escape(s) + r"\b", low):
                ctx = low[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                if s == "ag" + "ent" and (
                    ("pursuing " + s) in ctx or (("the " + s) in ctx and "simulat" in ctx)
                ):
                    continue
                failures.append(f"{rel}: forbidden string {s!r}: ...{ctx}...")
        for w in WARN_WORDS:
            if w in low:
                warnings.append(f"{rel}: discouraged word {w!r}")
    for w in warnings:
        print("WARN", w)
    for f in failures:
        print("FAIL", f)
    print(f"check: {len(failures)} failures, {len(warnings)} warnings over {len(files)} public files")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    p = sub.add_parser("data")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.add_argument("--verify", action="store_true", help="only verify hashes of files already present")
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("audit")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.set_defaults(func=cmd_audit)
    sub.add_parser("simulate").set_defaults(func=cmd_simulate)
    p = sub.add_parser("recover")
    p.add_argument(
        "--max-conditions", type=int, default=None, help="limit the number of conditions (smoke runs)"
    )
    p.set_defaults(func=cmd_recover)
    sub.add_parser("figures").set_defaults(func=cmd_figures)
    sub.add_parser("tables").set_defaults(func=cmd_tables)
    p = sub.add_parser("fit")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.add_argument("--null-b", type=int, default=None, help="override the parametric-null replicate count")
    p.add_argument("--jobs", type=int, default=1, help="processes for the parametric-null replicates")
    p.add_argument("--reuse-null", action="store_true", help="keep the null replicates of the previous run")
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("followup")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.add_argument("--reuse-null", action="store_true", help="keep the null replicates of the previous run")
    p.set_defaults(func=cmd_followup)
    p = sub.add_parser("sensitivity")
    p.add_argument("which", nargs="*", help="scale, loao, or both (default)")
    p.set_defaults(func=cmd_sensitivity)
    p = sub.add_parser("posteriors")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.add_argument(
        "--followup", action="store_true", help="the four-state follow-up instead of the frozen fits"
    )
    p.set_defaults(func=cmd_posteriors)
    p = sub.add_parser("evaluate")
    p.add_argument("datasets", nargs="*", default=["michaiel2020", "holmgren2021", "mabe22"])
    p.set_defaults(func=cmd_evaluate)
    sub.add_parser("animate").set_defaults(func=cmd_animate)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    p = sub.add_parser("paper")
    p.add_argument("--no-pdf", action="store_true", help="write the numbers and tables only")
    p.set_defaults(func=cmd_paper)
    sub.add_parser("test").set_defaults(func=cmd_test)
    sub.add_parser("lint").set_defaults(func=cmd_lint)
    sub.add_parser("check").set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
