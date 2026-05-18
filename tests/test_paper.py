"""Consistency of the paper with the result files: the numbers and tables committed under paper/
must equal what the build regenerates from results/."""

from pathlib import Path

import pytest

from aor import paper

ROOT = Path(__file__).resolve().parents[1]


def _ready() -> bool:
    return (paper.PAPER / "numbers.tex").exists() and any((paper.PAPER / "tables").glob("*.tex"))


@pytest.mark.skipif(
    not _ready(), reason="paper/numbers.tex not built yet (run `python tasks.py paper --no-pdf`)"
)
def test_numbers_match_result_files():
    committed = (paper.PAPER / "numbers.tex").read_text(encoding="utf-8")
    regenerated = paper.collect().tex()
    assert committed == regenerated, (
        "paper/numbers.tex differs from the result files; rerun `python tasks.py paper`"
    )


@pytest.mark.skipif(not _ready(), reason="paper tables not built yet")
def test_tables_match_result_files(tmp_path):
    written = paper.write_tables(tmp_path)
    assert written, "no result tables found"
    for name in written:
        committed = (paper.PAPER / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        regenerated = (tmp_path / f"{name}.tex").read_text(encoding="utf-8")
        assert committed == regenerated, f"paper/tables/{name}.tex differs from results/tables/{name}.csv"


def test_macro_names_are_letters_only():
    nums = paper.Numbers()
    nums.add("MichGain", "0.1")
    with pytest.raises(AssertionError):
        nums.add("Mich2Gain", "0.1")


def test_fmt():
    assert paper.fmt(1234.0, 0, True) == "1,234"
    assert paper.fmt(0.12345, 3) == "0.123"
    assert paper.fmt(float("nan")) == "n/a"
    assert paper.fmt(None) == "n/a"


def test_readme_numbers_match_result_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in paper.readme_phrases():
        assert phrase in readme, f"README lacks the phrase built from the result files: {phrase!r}"


def test_mabe22_licence_notices():
    """Every folder of MABe22-derived results carries the CC BY-NC-SA notice, and the notices in
    mixed folders name files that exist."""
    for folder in ("results/fits/mabe22", "results/fits_persistence/mabe22", "results/evaluate/mabe22"):
        notice = ROOT / folder / "LICENSE"
        assert notice.exists() and "CC BY-NC-SA 4.0" in notice.read_text(encoding="utf-8"), folder
    for folder in ("results/audit", "results/tables", "results/figures", "results/animations"):
        notice = (ROOT / folder / "LICENSE-MABe22.md").read_text(encoding="utf-8")
        assert "CC BY-NC-SA 4.0" in notice
        listed = [line[2:].strip() for line in notice.splitlines() if line.startswith("- ")]
        assert listed and all((ROOT / folder / f).exists() for f in listed), folder
