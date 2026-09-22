"""Nothing in `vccp/` may be sized by hand.

`mini_data` deliberately uses a different gene count, target count and cell
count from the real release (CLAUDE.md §0), so a hardcoded size passes here
and breaks on the server — or, worse, silently truncates. This scans the
package's syntax tree for the literals that would do that.

`vccp/config.py` is exempt from the numeric scan, and only that scan: it is
the one module whose job is to hold defaults, and every value in it is a
choice the user can override. Nothing else gets to hold a number that came
from the data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Sizes of the real release and of mini_data. A module holding any of these
#: has baked in something it should have read from the data.
FORBIDDEN_INTS = {
    18533,  # challenge genes, real
    1983,  # challenge genes, mini
    978,  # LINCS landmarks
    955,  # panel genes, mini
    1028,  # rest genes, mini
    300,  # perturbations, real
    30,  # perturbations, mini
    400,  # cells per perturbation, real
    50,  # cells per perturbation, mini
    360000,  # submission rows, real
}

#: Context names are discovered from the filesystem, never listed in code.
FORBIDDEN_STRINGS = {"K562_gwps", "rpe1", "K562_essential"}

#: Modules exempt from the numeric scan only (see the module docstring).
NUMERIC_SCAN_EXEMPT = {"config.py"}

#: A line carrying this comment is exempt, for the cases where a banned
#: value is genuinely not a size — a percentile, say. Explicit and greppable
#: on purpose: an exemption should be something a reviewer can see and
#: disagree with, not something the scanner quietly allows.
EXEMPT_MARKER = "# not-a-size"


def _package_files() -> list[Path]:
    package = Path(__file__).resolve().parent.parent / "vccp"
    return sorted(package.rglob("*.py"))


def _offences(path: Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    scan_numbers = path.name not in NUMERIC_SCAN_EXEMPT
    found: list[str] = []

    def exempt(lineno: int) -> bool:
        return 0 < lineno <= len(lines) and EXEMPT_MARKER in lines[lineno - 1]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or exempt(node.lineno):
            continue
        value = node.value
        if scan_numbers and isinstance(value, int) and not isinstance(value, bool):
            if value in FORBIDDEN_INTS:
                found.append(f"{path.name}:{node.lineno}: literal {value}")
        elif isinstance(value, str) and value in FORBIDDEN_STRINGS:
            found.append(f"{path.name}:{node.lineno}: literal {value!r}")
    return found


def test_package_has_no_hardcoded_sizes():
    offences = [offence for path in _package_files() for offence in _offences(path)]
    assert not offences, (
        "these literals must come from the data, the manifest or the config:\n  "
        + "\n  ".join(offences)
    )


def _pool_offences(path: Path) -> list[str]:
    """Calls that size a pool by the machine rather than by the config.

    Parsed rather than grepped, so prose mentioning `n_jobs=-1` — including
    this package's own docstrings — is not an offence.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
            node.func, "id", None
        )
        if name == "cpu_count":
            found.append(f"{path.name}:{node.lineno}: cpu_count()")
        for keyword in node.keywords:
            if keyword.arg in ("n_jobs", "num_workers", "n_threads"):
                value = keyword.value
                negative = (
                    isinstance(value, ast.UnaryOp)
                    and isinstance(value.op, ast.USub)
                    and isinstance(value.operand, ast.Constant)
                )
                if negative:
                    found.append(f"{path.name}:{node.lineno}: {keyword.arg}=-1")
    return found


def test_package_never_sizes_a_pool_by_the_machine():
    """Pool sizes come from `resources`, never from the host's core count
    (CLAUDE.md §1.2) — the server is shared."""
    offences = [offence for path in _package_files() for offence in _pool_offences(path)]
    assert not offences, "use resources.cpu_threads / resources.num_workers:\n  " + "\n  ".join(
        offences
    )


@pytest.mark.parametrize("value", sorted(FORBIDDEN_INTS))
def test_the_scanner_would_catch_each_forbidden_literal(value, tmp_path):
    """The scan is only worth having if it actually fires."""
    module = tmp_path / "offender.py"
    module.write_text(f"N_GENES = {value}\n")
    assert _offences(module), f"the scanner missed {value}"


def test_the_exemption_marker_works_and_is_line_scoped(tmp_path):
    module = tmp_path / "marked.py"
    module.write_text(
        "PERCENTILES = [50]  # not-a-size: percentiles, not cell counts\n"
        "N_GENES = 18533\n"
    )
    offences = _offences(module)
    assert len(offences) == 1
    assert "18533" in offences[0]


def test_exemptions_are_few_and_explained():
    """An exemption is a claim; it should be rare and carry its reason."""
    marked = []
    for path in _package_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if EXEMPT_MARKER in line:
                marked.append((f"{path.name}:{lineno}", line.strip()))

    assert len(marked) <= 3, f"too many size-scan exemptions: {marked}"
    for where, line in marked:
        reason = line.split(EXEMPT_MARKER, 1)[1].strip()
        assert reason.startswith(":") and len(reason) > 10, f"{where} has no reason"
