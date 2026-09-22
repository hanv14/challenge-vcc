"""Prior block 7 — protein language model embeddings. Optional.

Built only when `priors.plm_path` points at a file that already exists. The
pipeline never downloads anything at runtime (CLAUDE.md §4.1, §7), so an
unset or missing path means the block simply does not exist — not an error,
and not a silent zero block either: `coverage.csv` records that it was not
configured.

Accepted formats, both keyed by challenge gene symbol:

* `.npz` with arrays `gene_symbols` (n,) and `embeddings` (n, d)
* `.csv` / `.tsv` whose first column is the symbol and the rest are numeric
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .blocks import FEATURE, TARGET, BlockInputs, BlockResult
from .decompose import resident_pca


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        missing = [key for key in ("gene_symbols", "embeddings") if key not in loaded]
        if missing:
            raise ValueError(f"{path}: missing array(s) {missing}")
        return [str(s) for s in loaded["gene_symbols"]], np.asarray(
            loaded["embeddings"], dtype=np.float32
        )

    separator = "\t" if path.suffix in (".tsv", ".txt") else ","
    frame = pd.read_csv(path, sep=separator)
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: expected a symbol column and at least one value column")
    symbols = frame.iloc[:, 0].astype(str).tolist()
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    return symbols, values


class ProteinLanguageModel:
    """Block 7 — sequence-derived embeddings, if the user supplies a file."""

    name = "plm"
    roles = (FEATURE, TARGET)
    reads_responses = False

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        configured = inputs.cfg.priors.plm_path
        if not configured:
            return []

        path = Path(configured)
        if not path.is_absolute() and inputs.cfg.repo_root is not None:
            path = inputs.cfg.repo_root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"priors.plm_path points at {path}, which does not exist. "
                "The pipeline never downloads embeddings; supply the file or "
                "unset priors.plm_path."
            )

        symbols, embeddings = _load_embeddings(path)
        index = inputs.index

        rows, positions = [], []
        for row, symbol in enumerate(symbols):
            position = index.get(symbol)
            if position is not None:
                rows.append(row)
                positions.append(position)
        if len(positions) < 2:
            raise ValueError(
                f"{path}: only {len(positions)} of its symbols are challenge genes"
            )

        present = embeddings[rows]
        width = inputs.cfg.priors.block_dim
        if present.shape[1] > width:
            components, variance_ratio = resident_pca(present.T, width, rng=inputs.rng)
            reduced = components
            explained = float(variance_ratio.sum())
        else:
            reduced = present
            explained = 1.0

        features = np.zeros((inputs.n_genes, reduced.shape[1]), dtype=np.float32)
        covered = np.zeros(inputs.n_genes, dtype=bool)
        features[positions] = reduced
        covered[positions] = True
        covered = inputs.apply_gene_restriction(covered, source=self.name)

        return [
            BlockResult(
                name=self.name,
                features=features,
                covered=covered,
                detail={
                    "source": str(path),
                    "n_symbols_in_file": len(symbols),
                    "n_matched": len(positions),
                    "input_dim": int(embeddings.shape[1]),
                    "explained_variance_ratio": explained,
                },
            )
        ]
