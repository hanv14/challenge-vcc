"""Typed configuration.

One config object drives every stage. Nothing in the pipeline is sized by
hand: gene counts, perturbation counts, cell counts and context names all
come from the data, and everything that *is* a choice (budgets, widths,
chunk sizes, thresholds) comes from here.

Unknown keys are rejected rather than ignored, so a typo in a YAML file
fails at load instead of silently leaving a default in place.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


#: The directory name that marks the tiny example rather than the real data.
MINI_DATA_DIR = "mini_data"


class ConfigError(ValueError):
    """A config file is malformed, or names a key the schema does not have."""


@dataclass(frozen=True)
class Resources:
    """Limits that keep the pipeline a good neighbour on the shared server.

    Every worker pool in the pipeline is sized from ``cpu_threads`` or
    ``num_workers``; nothing calls ``os.cpu_count()`` or passes ``n_jobs=-1``.
    """

    cpu_threads: int = 4
    num_workers: int = 2
    gpu_memory_fraction: float = 0.5
    max_ram_gb: float = 64.0

    def validate(self) -> None:
        if self.cpu_threads < 1:
            raise ConfigError("resources.cpu_threads must be at least 1")
        if self.num_workers < 0:
            raise ConfigError("resources.num_workers must not be negative")
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ConfigError("resources.gpu_memory_fraction must be in (0, 1]")
        if self.max_ram_gb <= 0:
            raise ConfigError("resources.max_ram_gb must be positive")


@dataclass(frozen=True)
class Priors:
    """Gene vocabulary and prior blocks (CLAUDE.md §4.1)."""

    #: Each prior block is reduced to this many components before it is
    #: standardized and concatenated.
    block_dim: int = 32
    #: Cells sampled per source when estimating a co-expression block. The
    #: estimate is over genes, so more cells buy accuracy, not coverage.
    coexpr_max_cells: int = 20000
    #: Power iterations in the randomized SVD. More is more accurate and
    #: costs one streaming pass over the cells each.
    coexpr_n_iter: int = 2
    #: Gene-group families smaller than this carry no usable signal.
    hgnc_min_group_size: int = 2
    #: A knockdown's response is z-scored per target before the direction
    #: decomposition, which makes a near-null response look like a confident
    #: random direction. These two keys are how that is handled: the floor is
    #: a fraction of the sub-block's median response magnitude, below which a
    #: target is reported as low-magnitude, and, when weighting is on, the
    #: point at which its influence on the basis is halved.
    phenotype_magnitude_floor: float = 0.5
    #: Learn the direction basis from the targets that actually responded,
    #: then project every target into it. Off means every target, however
    #: null, shapes the basis equally.
    phenotype_weight_by_magnitude: bool = True
    #: Protein language model embeddings (block 7). Built only when this
    #: points at a file that exists; never downloaded (CLAUDE.md §4.1).
    plm_path: str | None = None
    #: Rows read at a time when streaming cells for a co-expression block.
    cell_block_size: int = 2048
    #: Nearest neighbours examined by the prior checks.
    check_n_neighbors: int = 10
    #: Targets sampled for the response-correlation check. The check is over
    #: pairs, so this caps a quadratic; the server has thousands of targets.
    check_max_targets: int = 500
    #: Gene-group name fragments the neighbour check looks for, matched
    #: case-insensitively against HGNC `gene_group`.
    check_families: tuple[str, ...] = (
        "ribosomal",
        "proteasome",
        "mitochondrial complex",
        "mitochondrial respiratory chain",
    )

    def validate(self) -> None:
        if self.block_dim < 1:
            raise ConfigError("priors.block_dim must be at least 1")
        if self.coexpr_max_cells < 2:
            raise ConfigError("priors.coexpr_max_cells must be at least 2")
        if self.coexpr_n_iter < 0:
            raise ConfigError("priors.coexpr_n_iter must not be negative")
        if self.cell_block_size < 1:
            raise ConfigError("priors.cell_block_size must be at least 1")
        if self.check_n_neighbors < 1:
            raise ConfigError("priors.check_n_neighbors must be at least 1")
        if self.check_max_targets < 2:
            raise ConfigError("priors.check_max_targets must be at least 2")
        if self.phenotype_magnitude_floor <= 0:
            raise ConfigError("priors.phenotype_magnitude_floor must be positive")


@dataclass(frozen=True)
class Model:
    """The shared gene-token model (CLAUDE.md §4.2)."""

    #: Width of a gene embedding: `W . prior(g) + delta(g)`.
    embedding_dim: int = 128
    #: L2 pull on the free per-gene vector, which starts at zero.
    delta_l2: float = 1e-4
    #: Latent bottleneck size. Cost does not grow with the number of output
    #: genes, so this is the main capacity knob.
    n_latents: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    #: Output genes decoded at a time. Lower this first if the GPU runs out.
    gene_chunk: int = 4096
    #: Rank of the per-phase and per-context LoRA adapters (§4.3).
    adapter_rank: int = 8

    def validate(self) -> None:
        if self.embedding_dim < 1:
            raise ConfigError("model.embedding_dim must be at least 1")
        if self.delta_l2 < 0:
            raise ConfigError("model.delta_l2 must not be negative")
        if self.embedding_dim % self.n_heads:
            raise ConfigError(
                f"model.embedding_dim ({self.embedding_dim}) must be divisible by "
                f"model.n_heads ({self.n_heads})"
            )
        for key in ("n_latents", "n_blocks", "n_heads", "gene_chunk", "adapter_rank"):
            if getattr(self, key) < 1:
                raise ConfigError(f"model.{key} must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class Train:
    """Budgets and the guard against forgetting (CLAUDE.md §4.3)."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    #: Mixed precision, when the device is CUDA. Ignored on CPU.
    amp: bool = True
    #: L2-SP: how hard a later phase is pulled toward the previous phase's
    #: weights. On by default in Phases 2 and 3 (§4.3).
    l2sp_weight: float = 1e-3
    #: Fraction of a later phase's steps that replay an earlier phase's
    #: batches instead.
    replay_fraction: float = 0.25
    #: Train a second Phase 2 arm with the core frozen, and score Phase 1
    #: validation under both, so the `unfreeze_core` default rests on a
    #: measurement rather than on a judgement (PLAN.md §8.9).
    core_freeze_ablation: bool = True
    #: Train a third Phase 2 arm with **no Phase 1 input at all** — no warm
    #: start from its checkpoint and no replay of its objective — so that
    #: what the first phase contributes is a measurement rather than an
    #: assumption (PLAN_PERCELL.md §7.6). The arm gets the full step budget
    #: on Phase 2's own objective where the warm arms spend
    #: `replay_fraction` of theirs on Phase 1's, which biases the comparison
    #: *against* Phase 1; `n_phase2_steps` is reported per arm so the
    #: comparison can be read with that in view.
    phase1_contribution_ablation: bool = True
    #: The ablation arm's budget, as a fraction of the main arm's.
    ablation_steps_fraction: float = 1.0
    #: L2 pull on each assay's perturbation-type offset, which starts at
    #: zero. It is what keeps a knockout row and a knockdown row near their
    #: shared base instead of each phase training its own in isolation
    #: (PLAN_PERCELL.md §7.4).
    type_delta_l2: float = 1e-3
    #: Steps between logged curve points.
    log_every: int = 25
    #: Ask torch for deterministic kernels, turn TF32 off, and fix the cuBLAS
    #: workspace. Two runs of the same checkpoint with the same seeds then
    #: produce the same predictions. Without it they do not: a CUDA matmul
    #: reduces in whatever order it likes, which moved thousands of genes
    #: across the generator's confidence threshold between two runs of the
    #: same submission (DECISIONS.md D49). It costs some speed, and a run
    #: that is not reproducible cannot be optimized against.
    deterministic: bool = True
    #: Output genes decoded per training step, sampled fresh each time. The
    #: loss is a mean over genes, so a subsample is an unbiased estimate of
    #: it, and the decoder's cost is linear in this. 0 means every gene —
    #: which on the server would be 18,533 of them per step. Evaluation
    #: always uses every gene.
    output_genes_per_step: int = 2048

    def validate(self) -> None:
        if self.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if self.l2sp_weight < 0:
            raise ConfigError("train.l2sp_weight must not be negative")
        if not 0.0 <= self.replay_fraction < 1.0:
            raise ConfigError("train.replay_fraction must be in [0, 1)")
        if not 0.0 < self.ablation_steps_fraction <= 1.0:
            raise ConfigError("train.ablation_steps_fraction must be in (0, 1]")
        if self.type_delta_l2 < 0:
            raise ConfigError("train.type_delta_l2 must not be negative")
        if self.log_every < 1:
            raise ConfigError("train.log_every must be at least 1")
        if self.output_genes_per_step < 0:
            raise ConfigError("train.output_genes_per_step must not be negative")


@dataclass(frozen=True)
class Phase1:
    """LINCS: control + perturbation -> signature -> perturbed (§4.4)."""

    steps: int = 400
    batch_size: int = 32
    #: Held-out fraction of *target genes*, so validation is unseen
    #: perturbations rather than unseen rows.
    val_fraction: float = 0.2
    loss_sig: float = 1.0
    loss_gmt: float = 0.5
    loss_pert: float = 1.0
    #: Feed the *true* signature into the second step instead of the
    #: predicted one. Off by default: the two steps train jointly.
    teacher_forcing: bool = False
    #: The assay's perturbation modality. LINCS is CRISPR knockout, and the
    #: challenge is CRISPR interference — §3.3 says directions transfer
    #: between them but magnitudes do not, so they get separate rows of the
    #: type vocabulary around a shared base (PLAN_PERCELL.md §7.4).
    pert_type: str = "crispr_ko"

    def validate(self) -> None:
        _check_phase("phase1", self.steps, self.batch_size, self.val_fraction)


@dataclass(frozen=True)
class Phase2:
    """Replogle: Siamese panel->rest, plus the perturbation module (§4.5)."""

    steps: int = 600
    batch_size: int = 32
    #: Cells drawn per target when sampling a batch.
    cells_per_draw: int = 16
    val_fraction: float = 0.2
    loss_mapping: float = 1.0
    loss_perturbation: float = 1.0
    #: The delta consistency term: the difference between the mapping's two
    #: arms has to equal the change that was observed (PLAN_PERCELL.md §5).
    #: Each arm's own MSE is dominated by the baseline expression level, so
    #: without this the mapping is never asked about the one quantity the six
    #: official metrics score.
    loss_delta: float = 0.5
    #: Cells per draw when *scoring* the delta. Larger than `cells_per_draw`
    #: on purpose: an observed change is a difference of two sampled means,
    #: and its noise falls as 1/n, so a small draw would report a ratio near
    #: 1.0 even for a perfect predictor. `map_delta_noise_floor_ratio` says
    #: where the floor actually landed.
    delta_eval_cells: int = 128
    #: Held-out targets the delta is scored over.
    delta_eval_targets: int = 8
    #: How the perturbation module is supervised (PLAN_PERCELL.md §3).
    #:
    #: * `pooled` — one control pseudobulk in, the target's pseudobulk as
    #:   truth. One mean effect per (context, target).
    #: * `percell` — a **cell** in, that cell perturbed out, with the truth
    #:   built by optimal transport between a drawn control batch and a drawn
    #:   perturbed batch.
    #:
    #: The default stays `pooled` until the rehearsal measures which wins
    #: (P5): the specification asks for per-cell, but a default that has not
    #: been measured is a guess, and `pooled` is the arm with a leaderboard
    #: number behind it.
    perturbation_mode: str = "pooled"
    #: OT regularization, as a **fraction of the batch's mean cost**
    #: (DECISIONS.md D52). The pooled-to-hard transition sits between 0.005
    #: and 0.05; `inf` recovers the pooled target exactly.
    #:
    #: 0.02 rather than something smaller because two measured quantities
    #: pull opposite ways (DECISIONS.md D61). A small value gives each cell a
    #: distinct target, which is the point; but a target averaged over few
    #: cells carries its own sampling noise, and below about 0.02 that noise
    #: exceeds what the pairing removed — the paired target ends up *further*
    #: from its control cell than the pooled mean is. 0.02 is where the
    #: coupling measurably stops adding noise while still giving the targets
    #: real per-cell variation. P5 sweeps it.
    ot_epsilon: float = 0.02
    #: Cells drawn per side, per target, for the coupling. The main cost and
    #: memory knob of `percell` mode: the perturbation module sees
    #: `ot_batch_cells x ot_targets_per_step` rows per step where `pooled`
    #: sees `batch_size`.
    ot_batch_cells: int = 64
    #: Targets per step in `percell` mode. Lower than `pooled`'s
    #: `batch_size`, because each target now costs a coupling and a batch of
    #: cells rather than one pseudobulk row.
    ot_targets_per_step: int = 2
    #: Sinkhorn iterations. A ceiling: the solver stops once the coupling's
    #: rows converge, which takes a handful of iterations at `ot_epsilon`
    #: 0.05 and a few dozen at 0.01.
    ot_iterations: int = 200
    #: Feed the delta term's decoder the model's *predicted* perturbed panel
    #: rather than the OT-paired true one (PLAN_PERCELL.md §5). Off by
    #: default: the true panel trains the mapping on changes without letting
    #: a bad perturbation prediction corrupt it.
    delta_through_prediction: bool = False
    #: Replogle is CRISPR interference, as the challenge is — the two share
    #: a type row, which is the transfer this vocabulary exists to allow.
    pert_type: str = "crispri"
    #: Phase 2 has the data to support training the core, and is where the
    #: forgetting guard is worth having (DECISIONS.md D3).
    unfreeze_core: bool = True

    def validate(self) -> None:
        _check_phase("phase2", self.steps, self.batch_size, self.val_fraction)
        if self.cells_per_draw < 1:
            raise ConfigError("phase2.cells_per_draw must be at least 1")
        if self.loss_delta < 0:
            raise ConfigError("phase2.loss_delta must not be negative")
        if self.delta_eval_cells < 2:
            raise ConfigError("phase2.delta_eval_cells must be at least 2")
        if self.delta_eval_targets < 1:
            raise ConfigError("phase2.delta_eval_targets must be at least 1")
        if self.perturbation_mode not in ("pooled", "percell"):
            raise ConfigError(
                "phase2.perturbation_mode must be 'pooled' or 'percell', got "
                f"{self.perturbation_mode!r}"
            )
        if not self.ot_epsilon > 0:
            raise ConfigError(
                "phase2.ot_epsilon must be positive; the hard-assignment limit is "
                "approached with a small positive value, not with zero"
            )
        if self.ot_batch_cells < 2:
            raise ConfigError("phase2.ot_batch_cells must be at least 2")
        if self.ot_targets_per_step < 1:
            raise ConfigError("phase2.ot_targets_per_step must be at least 1")
        if self.ot_iterations < 1:
            raise ConfigError("phase2.ot_iterations must be at least 1")


def _check_phase(name: str, steps: int, batch_size: int, val_fraction: float) -> None:
    if steps < 1:
        raise ConfigError(f"{name}.steps must be at least 1")
    if batch_size < 1:
        raise ConfigError(f"{name}.batch_size must be at least 1")
    if not 0.0 < val_fraction < 1.0:
        raise ConfigError(f"{name}.val_fraction must be in (0, 1)")


@dataclass(frozen=True)
class Phase3:
    """Adapting on the challenge controls, and predicting (CLAUDE.md §4.7)."""

    #: Steps of controls-only adaptation per challenge context.
    adapt_steps: int = 300
    batch_size: int = 32
    #: A challenge context arrives as control cells and nothing else, so the
    #: core stays frozen here whatever Phase 2 did (DECISIONS.md D3).
    unfreeze_core: bool = False
    #: The challenge is CRISPR interference, the same modality as Replogle,
    #: so Phase 3 reuses Phase 2's type row rather than starting a new one.
    #: The final test round is the same assay; a genuinely different one
    #: would set a different value here and get its own row.
    pert_type: str = "crispri"
    #: Where the knockdown prior's fold_expr comes from when the target is
    #: not in any Replogle bulk file.
    knockdown_source: str = "pooled"
    #: The prior is applied only where the target gene is detectably
    #: expressed in that context's controls (DECISIONS.md D8). Same floor as
    #: sanity check 3.
    knockdown_min_control_cpm: float = 1.0

    def validate(self) -> None:
        if self.adapt_steps < 1:
            raise ConfigError("phase3.adapt_steps must be at least 1")
        if self.batch_size < 1:
            raise ConfigError("phase3.batch_size must be at least 1")
        if self.knockdown_source not in ("pooled", "per_target_only"):
            raise ConfigError(
                "phase3.knockdown_source must be 'pooled' or 'per_target_only'"
            )
        if self.knockdown_min_control_cpm < 0:
            raise ConfigError("phase3.knockdown_min_control_cpm must not be negative")


@dataclass(frozen=True)
class Predict:
    """Generating the submitted cells (CLAUDE.md §4.7)."""

    #: The generator is a swappable component; this names it.
    generator: str = "count_space"
    #: Targets predicted before the block is flushed to disk.
    target_chunk: int = 1
    #: Control cells held in memory per context while generating.
    max_control_cells: int = 0
    #: Cells pushed through the model at once in `percell` mode. The
    #: encoder's cost is linear in the number of rows, so `cells_per_pert`
    #: rows over the whole panel at once is the shape most likely to exhaust
    #: a GPU; lower this first if it does.
    cells_per_forward: int = 64
    #: The expression level, in CP10K, below which a gene is not
    #: distinguishable from absent. It is the floor of every log fold change
    #: the pipeline forms, so that "this gene goes off" is a finite statement
    #: rather than a number set by whichever epsilon guards the division: at
    #: 1e-6 a gene sitting at 0.05 CP10K reads as a 15-log2 drop when it is
    #: predicted to zero, which says more about the epsilon than about the
    #: prediction. One count in a 20,000-UMI cell is 0.5 CP10K, so this is
    #: fifty times below anything a single cell can show.
    cpm_floor: float = 0.01

    def validate(self) -> None:
        if self.cells_per_forward < 1:
            raise ConfigError("predict.cells_per_forward must be at least 1")
        if self.target_chunk < 1:
            raise ConfigError("predict.target_chunk must be at least 1")
        if self.cpm_floor <= 0:
            raise ConfigError("predict.cpm_floor must be positive")


@dataclass(frozen=True)
class Checks:
    """What `check-data` verifies beyond files, columns and gene orders."""

    #: Also *read* blocks of every matrix, so a truncated or undecodable
    #: `.h5ad` is found in seconds rather than hours into the priors. A file
    #: whose `obs` and `var` are perfect can still fail mid-run with an HDF5
    #: "filter returned failure during read".
    probe_matrices: bool = True
    #: Row blocks read from a file too large to read in full, and their size.
    probe_samples: int = 12
    probe_block_rows: int = 512

    def validate(self) -> None:
        if self.probe_samples < 1:
            raise ConfigError("checks.probe_samples must be at least 1")
        if self.probe_block_rows < 1:
            raise ConfigError("checks.probe_block_rows must be at least 1")


@dataclass(frozen=True)
class Submission:
    """The submission format (CLAUDE.md §8), whose limits are the challenge's.

    They are config keys rather than module literals so that a round with
    different limits needs no code change — and so that a test can shrink
    them to check that the validator enforces them at all.
    """

    #: No cell may total more than this many counts.
    max_counts_per_cell: int = 1_000_000
    #: Stored entries in the whole file, explicit zeros included.
    max_stored_entries: int = 4_750_000_000
    #: Rows appended to the on-disk sparse dataset at a time. The file is
    #: assembled block by block; nothing concatenates it in RAM (§8).
    write_chunk_cells: int = 4096

    def validate(self) -> None:
        if self.max_counts_per_cell < 1:
            raise ConfigError("submission.max_counts_per_cell must be at least 1")
        if self.max_stored_entries < 1:
            raise ConfigError("submission.max_stored_entries must be at least 1")
        if self.write_chunk_cells < 1:
            raise ConfigError("submission.write_chunk_cells must be at least 1")


@dataclass(frozen=True)
class Sanity:
    """What "reasonable" means, check by check (CLAUDE.md §6.1).

    Every threshold the seven checks use lives here, with the defaults the
    specification names.
    """

    # check 2 — plausible magnitudes
    library_ratio_low: float = 0.5
    library_ratio_high: float = 2.0
    #: How many standard errors of the control mean a gene the model did not
    #: predict to move may sit from it, in **counts**. Those cells are
    #: resampled control cells, so this is the sampling range their mean
    #: should stay inside.
    control_mean_sigmas: float = 6.0
    #: The other half of check 2 (PLAN.md §8.5): genes the model *did* move
    #: are bounded too, or the check leaves them free by construction.
    max_abs_log2fc: float = 10.0

    # check 3 — the knockdown is visible
    knockdown_min_control_cpm: float = 1.0
    knockdown_max_ratio: float = 0.7
    knockdown_min_fraction: float = 0.9

    # check 4 — targets differ
    max_median_target_correlation: float = 0.95

    # check 5 — cells vary
    variance_ratio_low: float = 0.5
    variance_ratio_high: float = 2.0

    # check 7 — a reasonable number of DE calls
    nsig_ratio_low: float = 0.25
    nsig_ratio_high: float = 4.0

    #: Worst offenders listed by name in the report.
    max_reported: int = 10
    #: Accept a warning-level result without `--allow-warnings`. §6.1 says
    #: warnings are expected on `mini_data`, where the statistics are noisy,
    #: so `mini.yaml` turns this on and `server.yaml` leaves it off — the
    #: gate stays real where the numbers mean something. A **fail** still
    #: stops the run either way.
    accept_warnings: bool = False

    def validate(self) -> None:
        if not 0 < self.library_ratio_low < self.library_ratio_high:
            raise ConfigError("sanity.library_ratio_low must be in (0, library_ratio_high)")
        if not 0 < self.variance_ratio_low < self.variance_ratio_high:
            raise ConfigError("sanity.variance_ratio_low must be in (0, variance_ratio_high)")
        if not 0 < self.nsig_ratio_low < self.nsig_ratio_high:
            raise ConfigError("sanity.nsig_ratio_low must be in (0, nsig_ratio_high)")
        if not 0 < self.knockdown_max_ratio <= 1:
            raise ConfigError("sanity.knockdown_max_ratio must be in (0, 1]")
        if not 0 < self.knockdown_min_fraction <= 1:
            raise ConfigError("sanity.knockdown_min_fraction must be in (0, 1]")
        if not 0 < self.max_median_target_correlation <= 1:
            raise ConfigError("sanity.max_median_target_correlation must be in (0, 1]")
        if self.max_abs_log2fc <= 0:
            raise ConfigError("sanity.max_abs_log2fc must be positive")
        if self.control_mean_sigmas <= 0:
            raise ConfigError("sanity.control_mean_sigmas must be positive")
        if self.max_reported < 1:
            raise ConfigError("sanity.max_reported must be at least 1")


@dataclass(frozen=True)
class Report:
    """The results summary for the proposal defense (CLAUDE.md §5)."""

    #: Figures are drawn for a projector: large fonts, labelled axes.
    figure_dpi: int = 150
    font_size: int = 14
    #: Targets drawn in the predicted-vs-true scatter panel.
    max_scatter_targets: int = 4

    def validate(self) -> None:
        if self.figure_dpi < 50:
            raise ConfigError("report.figure_dpi must be at least 50")
        if self.font_size < 6:
            raise ConfigError("report.font_size must be at least 6")
        if self.max_scatter_targets < 1:
            raise ConfigError("report.max_scatter_targets must be at least 1")


@dataclass(frozen=True)
class Rehearsal:
    """Phase 3's procedure, run where the answers are known (CLAUDE.md §4.6)."""

    #: Held-out targets scored per variant. Capped so DE scoring stays fast.
    max_targets: int = 100
    #: Steps for the controls-only adaptation each variant runs.
    adapt_steps: int = 150
    #: Steps for the Phase 2 arms the rehearsal retrains under its own scopes.
    phase2_steps: int = 300
    #: Real cells kept per target when scoring (0 = all of them).
    cells_per_target: int = 0
    #: Control cells sampled for the scoring side (0 = all of them).
    max_control_cells: int = 2000
    #: Fraction of a screen's rest genes hidden by variant 3.
    unseen_gene_fraction: float = 0.1
    #: Screens variant 3 runs on. It retrains Phase 2 per screen, so one is
    #: usually enough to see whether the answer is positive.
    unseen_genes_contexts: int = 1
    #: Per-target predicted fold changes stored in the report, for the
    #: predicted-vs-true scatter plots of §5.
    n_saved_predictions: int = 5
    #: Grid searched for the generator's two calibration settings (§4.7).
    calibration_thresholds: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25)
    calibration_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    #: Fit the generator's two settings on **every** scorable screen rather
    #: than only the one with the most targets, and report the spread
    #: (PLAN_PERCELL.md §7.5). One screen's fit carried to the challenge is a
    #: cross-assay assumption; fitting it on several says whether it holds.
    #: Off by default because each screen costs a full grid of official
    #: scorings, which is the slow part of the rehearsal.
    calibrate_per_dataset: bool = False
    #: Learn what a perturbation does from LINCS alone, adapt to a Replogle
    #: screen through its control cells, and score (PLAN_PERCELL.md §7.7).
    #: Variant 2 is cross-*context*, the same assay in a different cell line;
    #: this is cross-*assay*, which is what every submission actually does —
    #: the challenge is a different lab, protocol and sequencing depth from
    #: Replogle. It is evidence for the write-up rather than score for the
    #: leaderboard, so it is off by default and never runs in a deadline run.
    cross_assay: bool = False
    #: Screens it runs on. It retrains nothing, but each screen costs an
    #: adaptation, a prior rebuild and three scored arms.
    cross_assay_contexts: int = 1
    #: The A/B that decides `phase2.perturbation_mode` (PLAN_PERCELL.md §12).
    #: Off by default because it retrains one Phase 2 arm per configuration:
    #: it is a deliberate measurement, not something every run should pay for.
    mode_sweep: bool = False
    #: The configurations it compares. `None` is the pooled arm — the control
    #: the per-cell mode has to beat — and each number is `phase2.ot_epsilon`
    #: as a fraction of the batch's mean cost. The grid spans the window
    #: where the coupling actually changes: below ~0.005 nothing converges,
    #: and by 0.05 the coupling is already near-uniform (DECISIONS.md D52,
    #: D61).
    mode_sweep_epsilons: tuple[float | None, ...] = (None, 0.005, 0.01, 0.02, 0.05)
    #: Screens the sweep runs on. One by default: each screen multiplies the
    #: cost by the length of the grid, and §14's budget is five runs rather
    #: than ninety.
    mode_sweep_contexts: int = 1

    def validate(self) -> None:
        if self.max_targets < 2:
            raise ConfigError("rehearsal.max_targets must be at least 2")
        for key in ("adapt_steps", "phase2_steps"):
            if getattr(self, key) < 1:
                raise ConfigError(f"rehearsal.{key} must be at least 1")
        if not 0.0 < self.unseen_gene_fraction < 1.0:
            raise ConfigError("rehearsal.unseen_gene_fraction must be in (0, 1)")
        if not self.calibration_thresholds or not self.calibration_scales:
            raise ConfigError("rehearsal calibration grids must not be empty")
        if any(t < 0 for t in self.calibration_thresholds):
            raise ConfigError("rehearsal.calibration_thresholds must not be negative")
        if any(s <= 0 for s in self.calibration_scales):
            raise ConfigError("rehearsal.calibration_scales must be positive")
        if not self.mode_sweep_epsilons:
            raise ConfigError("rehearsal.mode_sweep_epsilons must not be empty")
        if any(e is not None and not e > 0 for e in self.mode_sweep_epsilons):
            raise ConfigError(
                "rehearsal.mode_sweep_epsilons must be positive, or null for the "
                "pooled arm"
            )
        if self.mode_sweep_contexts < 1:
            raise ConfigError("rehearsal.mode_sweep_contexts must be at least 1")
        if self.cross_assay_contexts < 1:
            raise ConfigError("rehearsal.cross_assay_contexts must be at least 1")


@dataclass(frozen=True)
class Eval:
    """The official scorer (CLAUDE.md §6)."""

    #: Pinned explicitly: the preset ships `auto`, and which DE engine a host
    #: happens to have would otherwise change the numbers (§1).
    de_backend: str = "pdex"
    #: Build the leaderboard's 0 = baseline / 1 = replicate scale where the
    #: data allow it. Reference points of the scorer only — never predictions,
    #: never submitted, never used to train or choose the model (§6).
    build_scale: bool = True

    def validate(self) -> None:
        allowed = {"auto", "pdex", "scanpy", "deseq2", "gpudge"}
        if self.de_backend not in allowed:
            raise ConfigError(
                f"eval.de_backend must be one of {sorted(allowed)} (got {self.de_backend!r})"
            )


@dataclass(frozen=True)
class Config:
    """The whole configuration, as loaded from a YAML file.

    ``source`` is the file it came from; ``repo_root`` is what relative paths
    in that file resolve against (the repository root, not the process's
    working directory, so a run from elsewhere still finds ``mini_data``).
    """

    data_root: Path
    vcc_root: Path
    output_root: Path
    device: str = "auto"
    run_name: str = "run"
    seed: int = 0
    resources: Resources = field(default_factory=Resources)
    priors: Priors = field(default_factory=Priors)
    model: Model = field(default_factory=Model)
    train: Train = field(default_factory=Train)
    phase1: Phase1 = field(default_factory=Phase1)
    phase2: Phase2 = field(default_factory=Phase2)
    phase3: Phase3 = field(default_factory=Phase3)
    predict: Predict = field(default_factory=Predict)
    rehearsal: Rehearsal = field(default_factory=Rehearsal)
    eval: Eval = field(default_factory=Eval)
    checks: Checks = field(default_factory=Checks)
    submission: Submission = field(default_factory=Submission)
    sanity: Sanity = field(default_factory=Sanity)
    report: Report = field(default_factory=Report)

    source: Path | None = None
    repo_root: Path | None = None

    # ---- derived paths ---------------------------------------------------
    @property
    def ref_dir(self) -> Path:
        return self.data_root / "ref"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def phases_dir(self) -> Path:
        return self.processed_dir / "phases"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def on_mini_data(self) -> bool:
        """Whether this run reads the tiny example rather than the real data.

        Only ever used to caveat a *report*: every number a mini run produces
        says that the pipeline works, and none of them says how well the
        method does (CLAUDE.md §5). It never changes what the code does —
        §1.1 forbids a mini-specific code path.

        Matched on a whole path component, not as a substring: a server
        directory that merely happens to contain the letters "mini" is real
        data, and labelling its results "not indicative" at a defense would
        be worse than saying nothing.
        """
        return MINI_DATA_DIR in self.data_root.parts

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError(f"device must be one of auto, cpu, cuda (got {self.device!r})")
        if not self.run_name:
            raise ConfigError("run_name must not be empty")
        self.resources.validate()
        self.priors.validate()
        self.model.validate()
        self.train.validate()
        self.phase1.validate()
        self.phase2.validate()
        self.phase3.validate()
        self.predict.validate()
        self.rehearsal.validate()
        self.eval.validate()
        self.checks.validate()
        self.submission.validate()
        self.sanity.validate()
        self.report.validate()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _build_section(cls: type, given: dict[str, Any]):
    """Build a config section, keeping tuple fields tuples so the whole
    config stays hashable and comparable."""
    coerced = dict(given)
    for f in dataclasses.fields(cls):
        if f.name in coerced and isinstance(coerced[f.name], list):
            coerced[f.name] = tuple(coerced[f.name])
    return cls(**coerced)


def _reject_unknown(section: str, given: dict[str, Any], cls: type) -> None:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {section}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}"
        )


def load_config(path: str | Path, repo_root: str | Path | None = None) -> Config:
    """Load a YAML config, resolving relative paths against the repo root."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    base = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent

    sections = {}
    for name, cls in (
        ("resources", Resources),
        ("priors", Priors),
        ("model", Model),
        ("train", Train),
        ("phase1", Phase1),
        ("phase2", Phase2),
        ("phase3", Phase3),
        ("predict", Predict),
        ("rehearsal", Rehearsal),
        ("eval", Eval),
        ("checks", Checks),
        ("submission", Submission),
        ("sanity", Sanity),
        ("report", Report),
    ):
        section_raw = raw.pop(name, {}) or {}
        if not isinstance(section_raw, dict):
            raise ConfigError(f"{path}: {name!r} must be a mapping")
        _reject_unknown(name, section_raw, cls)
        sections[name] = _build_section(cls, section_raw)

    # `source` and `repo_root` are set by the loader, never by the file.
    for reserved in ("source", "repo_root"):
        if reserved in raw:
            raise ConfigError(f"{path}: '{reserved}' is set by the loader, not by the config")

    _reject_unknown(str(path), raw, Config)
    for required in ("data_root", "vcc_root", "output_root"):
        if required not in raw:
            raise ConfigError(f"{path}: missing required key '{required}'")

    cfg = Config(
        data_root=_resolve(raw.pop("data_root"), base),
        vcc_root=_resolve(raw.pop("vcc_root"), base),
        output_root=_resolve(raw.pop("output_root"), base),
        source=path.resolve(),
        repo_root=base,
        **sections,
        **raw,
    )
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """A plain dict of the config, for writing into the run directory."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            out[f.name] = str(value)
        elif dataclasses.is_dataclass(value):
            out[f.name] = dataclasses.asdict(value)
        else:
            out[f.name] = value
    return out
