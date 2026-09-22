"""Every path the pipeline reads or writes, derived from the config.

No module outside this one builds a path from a string literal. Input paths
follow the layout of CLAUDE.md §3; output paths follow the run directory of
PLAN.md §2. Context names and Replogle context names are discovered from the
filesystem, never listed here.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config

# The HGNC table is `hgnc_complete_set.txt` on the server and `hgnc_subset.txt`
# in mini_data; both have the same columns, so either is accepted.
HGNC_GLOB = "hgnc_*.txt"


class DataPaths:
    """Input files (CLAUDE.md §3). Nothing here is written to."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ---- reference -------------------------------------------------------
    @property
    def challenge_genes(self) -> Path:
        return self.cfg.ref_dir / "challenge_genes.tsv"

    @property
    def challenge_perts(self) -> Path:
        return self.cfg.ref_dir / "challenge_perts.txt"

    @property
    def hgnc(self) -> Path:
        """The HGNC table under whichever of its two names is present."""
        matches = sorted(self.cfg.ref_dir.glob(HGNC_GLOB))
        if not matches:
            raise FileNotFoundError(
                f"no HGNC table matching {HGNC_GLOB!r} in {self.cfg.ref_dir} "
                "(expected hgnc_complete_set.txt or hgnc_subset.txt)"
            )
        return matches[0]

    # ---- challenge release ----------------------------------------------
    @property
    def manifest(self) -> Path:
        return self.cfg.vcc_root / "manifest.json"

    @property
    def gene_names(self) -> Path:
        return self.cfg.vcc_root / "gene_names.csv"

    @property
    def pert_counts(self) -> Path:
        return self.cfg.vcc_root / "pert_counts.csv"

    def context_h5ad(self, context: str) -> Path:
        return self.cfg.vcc_root / f"context_{context}.h5ad"

    def discover_context_files(self) -> dict[str, Path]:
        """Context label -> release file, read from the file names."""
        return {
            path.stem[len("context_") :]: path
            for path in sorted(self.cfg.vcc_root.glob("context_*.h5ad"))
        }

    # ---- phase 1 ---------------------------------------------------------
    @property
    def phase1_lincs(self) -> Path:
        return self.cfg.phases_dir / "phase1_lincs.h5ad"

    # ---- phase 2 ---------------------------------------------------------
    @property
    def gene_split(self) -> Path:
        return self.cfg.phases_dir / "gene_split.csv"

    @property
    def phase2_dir(self) -> Path:
        return self.cfg.phases_dir / "phase2"

    def phase2_context_dir(self, context: str) -> Path:
        return self.phase2_dir / context

    def discover_phase2_contexts(self) -> dict[str, Path]:
        """Replogle context name -> directory. `K562_essential` may be present
        on the server and absent in mini_data; both are handled by looking."""
        if not self.phase2_dir.is_dir():
            return {}
        return {
            path.name: path
            for path in sorted(self.phase2_dir.iterdir())
            if path.is_dir() and (path / "meta.json").is_file()
        }

    # ---- replogle source -------------------------------------------------
    @property
    def replogle_dir(self) -> Path:
        return self.cfg.processed_dir / "replogle"

    def discover_replogle_bulk(self) -> dict[str, Path]:
        """`<name>_raw_bulk` / `<name>_normalized_bulk` -> file."""
        if not self.replogle_dir.is_dir():
            return {}
        return {path.stem: path for path in sorted(self.replogle_dir.glob("*_bulk.h5ad"))}

    def discover_replogle_singlecell(self) -> dict[str, Path]:
        if not self.replogle_dir.is_dir():
            return {}
        return {
            path.stem: path for path in sorted(self.replogle_dir.glob("*_raw_singlecell.h5ad"))
        }

    # ---- phase 3 ---------------------------------------------------------
    @property
    def phase3_dir(self) -> Path:
        return self.cfg.phases_dir / "phase3"

    @property
    def phase3_genes(self) -> Path:
        return self.phase3_dir / "genes.csv"

    @property
    def phase3_targets(self) -> Path:
        return self.phase3_dir / "targets.csv"

    def phase3_controls(self, context: str) -> Path:
        return self.phase3_dir / f"controls_{context}.h5ad"

    def discover_phase3_controls(self) -> dict[str, Path]:
        if not self.phase3_dir.is_dir():
            return {}
        return {
            path.stem[len("controls_") :]: path
            for path in sorted(self.phase3_dir.glob("controls_*.h5ad"))
        }

    # ---- coverage --------------------------------------------------------
    @property
    def pert_coverage(self) -> Path:
        return self.cfg.processed_dir / "pert_coverage.csv"

    @property
    def gene_coverage(self) -> Path:
        return self.cfg.processed_dir / "gene_coverage.csv"


class RunPaths:
    """Output files. Everything the pipeline writes lives under `run_dir`."""

    def __init__(self, cfg: Config) -> None:
        self.root = cfg.run_dir

    def ensure(self) -> None:
        for directory in (self.root, self.stages, self.reports, self.priors, self.checkpoints):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def log(self) -> Path:
        return self.root / "log.txt"

    @property
    def stages(self) -> Path:
        return self.root / ".stages"

    def stage_marker(self, stage: str) -> Path:
        return self.stages / f"{stage}.done"

    @property
    def priors(self) -> Path:
        return self.root / "priors"

    @property
    def prior_features(self) -> Path:
        return self.priors / "prior_features.npz"

    @property
    def prior_coverage(self) -> Path:
        return self.priors / "coverage.csv"

    @property
    def prior_checks(self) -> Path:
        return self.priors / "checks.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def core_checkpoint(self, phase: str) -> Path:
        return self.checkpoints / f"core_{phase}.pt"

    def phase_dir(self, phase: str) -> Path:
        return self.root / phase

    def phase_metrics(self, phase: str) -> Path:
        return self.phase_dir(phase) / "metrics.json"

    def phase_curves(self, phase: str) -> Path:
        return self.phase_dir(phase) / "curves.csv"

    @property
    def rehearsal_dir(self) -> Path:
        return self.root / "rehearsal"

    @property
    def rehearsal_report(self) -> Path:
        return self.rehearsal_dir / "report.json"

    @property
    def rehearsal_summary(self) -> Path:
        return self.rehearsal_dir / "summary.txt"

    def rehearsal_scale(self, variant: str, context: str) -> Path:
        """Where one dataset's leaderboard-scale bundle is written."""
        return self.rehearsal_dir / "scale" / f"{variant}_{context}"

    @property
    def phase3_policy(self) -> Path:
        return self.root / "phase3_policy.json"

    def phase3_adapt(self, context: str) -> Path:
        return self.phase_dir("phase3") / f"adapt_{context}.json"

    def phase3_knockdown(self, context: str) -> Path:
        return self.phase_dir("phase3") / f"knockdown_{context}.csv"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions" / "model"

    def prediction_block(self, context: str, target: str) -> Path:
        return self.predictions_dir / context / f"{target}.npz"

    @property
    def prediction_index(self) -> Path:
        return self.predictions_dir / "index.json"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def frozen_check(self) -> Path:
        return self.reports / "frozen_check.json"

    @property
    def forgetting(self) -> Path:
        return self.reports / "forgetting.json"

    @property
    def check_data_report(self) -> Path:
        return self.reports / "check_data.json"
