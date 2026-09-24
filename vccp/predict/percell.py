"""Per-cell prediction: these cells in, these cells perturbed out.

The shared core of `perturbation_mode: percell` (PLAN_PERCELL.md §3, §9).
Both the submission path (`predict/run.py`) and the rehearsal's A/B
(`rehearsal/variants.py`) need it, and they must do exactly the same thing:
an A/B whose two sides predict differently would be measuring the difference
between two prediction routines rather than between two trained models.

What it does, for one target:

* each drawn control cell goes through the perturbation module and the
  adapted mapping **on its own** — the model's head predicts a *change added
  to its input*, so a row comes back as its own baseline plus a predicted
  delta;
* the fold change is taken against **that same row**, because the generator
  multiplies that cell's own counts. Measuring against the population mean
  would apply the cell's own deviation from the mean a second time, and it
  would make the folds compensate for each cell's deviation until every cell
  landed on one profile — the collapse sanity check 5 exists to catch;
* a gene with no counts in a drawn cell keeps a fold change of exactly 1,
  because `0 x fold` is 0 whatever the fold. With a pooled baseline almost no
  gene sat at zero; with a per-cell baseline most do, so without this the
  reported effect sizes would be dominated by arithmetic about cells nothing
  can happen to.

Cells go through in chunks: the encoder's cost is linear in the number of
rows, and `cells_per_pert` rows over the whole panel at once is the shape
most likely to exhaust a GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from ..phases import adapt, phase2
from ..rehearsal import common


def control_sd_units(
    control_counts: np.ndarray, ctrl_mean: np.ndarray, ctrl_std: np.ndarray
) -> np.ndarray:
    """Raw counts to control-SD units, on the axis the counts came in on.

    `log1p_cp10k` is `data_prep`'s, reused rather than reimplemented
    (CLAUDE.md §2), so the units the rehearsal predicts in are the units
    Phase 2 trained in.
    """
    from data_prep.phase_data import log1p_cp10k

    counts = np.asarray(control_counts)
    library = np.maximum(counts.sum(axis=1), 1).astype(np.float64)
    lognorm = log1p_cp10k(counts.astype(np.float64), library)
    return ((lognorm - ctrl_mean) / ctrl_std).astype(np.float32)


def predict_cells(
    model,
    context,
    target_idx: torch.Tensor,
    baseline_sd: np.ndarray,
    panel_positions: np.ndarray,
    rest_positions: np.ndarray,
    device: str,
    pert_type: str | None = None,
    cells_per_forward: int = 64,
) -> np.ndarray:
    """`(n_cells, n_genes)` predicted profiles, in control-SD units.

    `context` is anything the model's two entry points accept — a Phase 2
    `ContextTensors` or a Phase 3 `ChallengeContext`; both carry the gene
    indices and the context profile those functions read.
    """
    n_cells = int(baseline_sd.shape[0])
    n_genes = int(baseline_sd.shape[1])
    panel_baseline = baseline_sd[:, panel_positions]
    predicted = np.zeros((n_cells, n_genes), dtype=np.float32)
    step = max(1, int(cells_per_forward))

    model.eval()
    with torch.no_grad():
        for start in range(0, n_cells, step):
            block = panel_baseline[start : start + step]
            values = torch.as_tensor(block, dtype=torch.float32, device=device)
            targets = target_idx.reshape(1).expand(values.shape[0])
            panel = phase2.predict_perturbed_panel(
                model, context, values, targets, pert_type
            )
            rest = adapt.map_panel_to_rest(model, context, panel)
            stop = start + values.shape[0]
            predicted[start:stop, panel_positions] = panel.cpu().numpy()
            predicted[start:stop, rest_positions] = rest.cpu().numpy()
    return predicted


def log2fc_for_cells(
    predicted_sd: np.ndarray,
    baseline_sd: np.ndarray,
    control_counts: np.ndarray,
    ctrl_mean: np.ndarray,
    ctrl_std: np.ndarray,
    cpm_floor: float,
) -> np.ndarray:
    """Predicted profiles to per-cell fold changes the generator can apply.

    `control_counts` is the same cells' raw counts, row for row: it is what
    says which entries a multiplicative generator cannot move.
    """
    log2fc = common.sd_units_to_log2fc(
        predicted_sd, ctrl_mean, ctrl_std, cpm_floor, baseline_sd=baseline_sd
    )
    log2fc[np.asarray(control_counts) == 0] = 0.0
    return log2fc
