"""The challenge release manifest (`vcc_root/manifest.json`).

Everything that would otherwise be a hardcoded number — the column names, the
control label, how many cells a perturbation needs, which contexts the round
has — is read from here (CLAUDE.md §3.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_KEYS = (
    "pert_col",
    "context_col",
    "control_label",
    "n_genes",
    "cells_per_pert",
    "per_context",
)


@dataclass(frozen=True)
class Manifest:
    pert_col: str
    context_col: str
    control_label: str
    n_genes: int
    cells_per_pert: int
    per_context: dict[str, dict[str, Any]]
    raw: dict[str, Any]

    @property
    def contexts(self) -> tuple[str, ...]:
        """Contexts of this round.

        `contexts` is the manifest's own list where it has one; otherwise the
        keys of `per_context`. Either way the round's contexts are data, so
        the final round's D/E/F need no code change.
        """
        listed = self.raw.get("contexts")
        if listed:
            return tuple(str(c) for c in listed)
        return tuple(self.per_context)

    def control_cells(self, context: str) -> int | None:
        return self.per_context.get(context, {}).get("control_cells")

    def ground_truth_cells(self, context: str) -> int | None:
        return self.per_context.get(context, {}).get("ground_truth_cells")


def load_manifest(manifest_path: Path) -> Manifest:
    raw = json.loads(manifest_path.read_text())
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{manifest_path}: missing key(s) {missing}")

    per_context = raw["per_context"]
    if not isinstance(per_context, dict) or not per_context:
        raise ValueError(f"{manifest_path}: 'per_context' must be a non-empty mapping")

    return Manifest(
        pert_col=str(raw["pert_col"]),
        context_col=str(raw["context_col"]),
        control_label=str(raw["control_label"]),
        n_genes=int(raw["n_genes"]),
        cells_per_pert=int(raw["cells_per_pert"]),
        per_context={str(k): dict(v) for k, v in per_context.items()},
        raw=raw,
    )
