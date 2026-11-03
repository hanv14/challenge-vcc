"""The final round's release check (FINAL_ROUND.md step 1, D136).

`data_prep` processes every `context_*.h5ad` it finds, so a validation file
left beside the final release would become a context of the final round.
The check must pass one clean release and refuse a mixed one.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys

import anndata as ad

from tests.conftest import REPO_ROOT

MINI = REPO_ROOT / "mini_data"


def load_script():
    path = REPO_ROOT / "scripts" / "final_release_check.py"
    spec = importlib.util.spec_from_file_location("final_release_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relabelled_release(root, rename=(("A", "D"), ("B", "E"), ("C", "F"))):
    rename = dict(rename)
    root.mkdir(parents=True)
    for old, new in rename.items():
        a = ad.read_h5ad(MINI / "vcc" / f"context_{old}.h5ad")
        a.obs["context"] = new
        a.write_h5ad(root / f"context_{new}.h5ad")
    manifest = json.loads((MINI / "vcc" / "manifest.json").read_text())
    manifest["contexts"] = [rename[c] for c in manifest["contexts"]]
    manifest["per_context"] = {rename[k]: v for k, v in manifest["per_context"].items()}
    manifest["partition"] = "test"
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name in ("gene_names.csv", "pert_counts.csv"):
        shutil.copy(MINI / "vcc" / name, root / name)
    return root


def test_one_clean_release_passes(tmp_path, capsys):
    release = relabelled_release(tmp_path / "final")
    assert load_script().main([str(release), "--validation-data", str(MINI)]) == 0
    out = capsys.readouterr().out
    assert "D, E, F" in out and "IDENTICAL" in out and "fast path" in out


def test_leftover_contexts_from_another_round_are_refused(tmp_path, capsys):
    release = relabelled_release(tmp_path / "mixed")
    shutil.copy(MINI / "vcc" / "context_A.h5ad", release / "context_A.h5ad")
    assert load_script().main([str(release), "--validation-data", str(MINI)]) == 1
    assert "context_A.h5ad is here but the manifest does not list it" in capsys.readouterr().out


def test_a_missing_context_or_file_is_refused(tmp_path, capsys):
    release = relabelled_release(tmp_path / "short")
    (release / "context_F.h5ad").unlink()
    assert load_script().main([str(release), "--validation-data", str(MINI)]) == 1
    assert "no " in capsys.readouterr().out
    (release / "pert_counts.csv").unlink()
    assert load_script().main([str(release), "--validation-data", str(MINI)]) == 1


def test_a_changed_gene_axis_sends_the_run_to_the_fallback(tmp_path, capsys):
    release = relabelled_release(tmp_path / "axis")
    lines = (release / "gene_names.csv").read_text().splitlines()
    (release / "gene_names.csv").write_text("\n".join(lines[:1] + lines[2:] + lines[1:2]) + "\n")
    assert load_script().main([str(release), "--validation-data", str(MINI)]) == 0
    assert "DIFFERENT" in capsys.readouterr().out
