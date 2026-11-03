#!/usr/bin/env python
"""Is this directory one clean challenge release, and can the validation
model predict it? (FINAL_ROUND.md step 1)

    python scripts/final_release_check.py /data/share/han/VCC_final \\
        --validation-data /data/han/projects/VCC/data

Reads only. Exits 1 on anything that would make the final run wrong:

* a file the pipeline reads is missing (manifest.json, gene_names.csv,
  pert_counts.csv, a context_<X>.h5ad for every context the manifest lists);
* a context file the manifest does not list (a leftover from another round:
  `data_prep` globs `context_*.h5ad`, so it would be processed as this
  round's);
* a context file whose `obs["context"]` is not its own label.

Then it prints what decides the path: the partition, the contexts,
`cells_per_pert`, the number of targets, how many of them the validation
round had, and whether the gene axis is the one the validation model was
trained on (if not, the model cannot be reused and the full `all` is needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import pandas as pd


def axis_of(gene_names: Path) -> list[str]:
    frame = pd.read_csv(gene_names)
    column = "gene_name" if "gene_name" in frame.columns else frame.columns[0]
    return frame[column].astype(str).tolist()


def context_labels(path: Path, column: str) -> set[str]:
    """The distinct values of obs[column], read without loading X."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if column not in obs:
            return set()
        node = obs[column]
        if isinstance(node, h5py.Group):                     # categorical
            categories = node["categories"][:]
            codes = set(node["codes"][:].tolist())
            return {categories[c].decode() if isinstance(categories[c], bytes)
                    else str(categories[c]) for c in codes if c >= 0}
        values = node[:]
        return {v.decode() if isinstance(v, bytes) else str(v) for v in set(values.tolist())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("release", type=Path, help="the unpacked release (vcc_root)")
    parser.add_argument("--validation-data", type=Path, required=True,
                        help="the validation round's data_root (its ref/challenge_genes.tsv "
                             "is the gene axis the validation model was trained on)")
    args = parser.parse_args(argv)
    release = args.release
    problems: list[str] = []

    for name in ("manifest.json", "gene_names.csv", "pert_counts.csv"):
        if not (release / name).is_file():
            problems.append(f"missing {release / name}")
    if problems:
        print("\n".join(f"FAIL  {p}" for p in problems))
        return 1

    manifest = json.loads((release / "manifest.json").read_text())
    contexts = [str(c) for c in manifest.get("contexts", sorted(manifest.get("per_context", {})))]
    present = sorted(p.stem.removeprefix("context_") for p in release.glob("context_*.h5ad"))
    for context in contexts:
        if context not in present:
            problems.append(f"the manifest lists context {context!r} but there is no "
                            f"{release / f'context_{context}.h5ad'}")
    for context in present:
        if context not in contexts:
            problems.append(f"context_{context}.h5ad is here but the manifest does not list it: "
                            "a leftover from another round, which data_prep would process as "
                            "this round's. Unpack the release into a directory of its own.")
    column = manifest.get("context_col", "context")
    for context in sorted(set(contexts) & set(present)):
        labels = context_labels(release / f"context_{context}.h5ad", column)
        if labels and labels != {context}:
            problems.append(f"context_{context}.h5ad has obs[{column!r}] = {sorted(labels)}")

    targets = pd.read_csv(release / "pert_counts.csv")
    pert_col = manifest.get("pert_col", "target_gene")
    final_targets = set(targets[pert_col].astype(str)) - {manifest.get("control_label", "")}
    axis = axis_of(release / "gene_names.csv")
    reference = args.validation_data / "ref" / "challenge_genes.tsv"
    validation_axis = pd.read_csv(reference, sep="\t", header=None)[0].astype(str).tolist()
    perts_file = args.validation_data / "ref" / "challenge_perts.txt"
    validation_targets = set(perts_file.read_text().split()) if perts_file.is_file() else set()

    print(f"release          : {release}")
    print(f"partition        : {manifest.get('partition')}   panel: {manifest.get('panel_id')}")
    print(f"contexts         : {', '.join(contexts)} (files: {', '.join(present)})")
    print(f"cells_per_pert   : {manifest.get('cells_per_pert')}")
    print(f"targets          : {len(final_targets)}, of which {len(final_targets & validation_targets)} "
          f"were validation-round targets")
    same_axis = axis == validation_axis
    print(f"gene axis        : {len(axis)} genes, "
          + ("IDENTICAL to the validation model's" if same_axis else
             f"DIFFERENT from the validation model's ({len(validation_axis)} genes)"))
    if not same_axis:
        print("  -> the validation model cannot be reused: take FINAL_ROUND.md's fallback "
              "(full `all`) instead of the fast path.")
    if problems:
        print("\n".join(f"FAIL  {p}" for p in problems))
        return 1
    print("OK    one clean release" + ("; the fast path applies" if same_axis else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
