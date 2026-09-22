"""The `priors` stage: build the vocabulary, check it, write the artifacts.

Produces checklist items 1–4 (CLAUDE.md §4.8):

    priors/prior_features.npz   both role priors, the layout, the scope
    priors/coverage.csv         one row per block: what it covered
    priors/checks.json          the §4.1 neighbour and response checks
"""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..logging_utils import get_logger
from ..paths import RunPaths
from .build import ROLES, build_priors
from .checks import run_checks
from .scope import FULL_SCOPE


def run_priors(cfg: Config) -> dict[str, Any]:
    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()

    # The submission's priors see everything; the rehearsal rebuilds them
    # under its own scope when it hides targets, contexts or genes (§4.1).
    features = build_priors(cfg, FULL_SCOPE)
    features.save(run_paths.prior_features)
    features.coverage.to_csv(run_paths.prior_coverage, index=False)

    checks = run_checks(cfg, features)
    run_paths.prior_checks.write_text(json.dumps(checks, indent=2, default=str))

    summary = {
        "n_genes": features.n_genes,
        "scope": features.scope.describe(),
        "roles": {
            role: {
                "width": features.width(role),
                "blocks": features.layouts[role].block_names,
            }
            for role in ROLES
        },
        "n_blocks_built": int(features.coverage["built"].sum()),
        "artifacts": {
            "prior_features": str(run_paths.prior_features),
            "coverage": str(run_paths.prior_coverage),
            "checks": str(run_paths.prior_checks),
        },
    }

    log.info(
        "priors written: feature-role width %d, target-role width %d  ->  %s",
        features.width("feature"),
        features.width("target"),
        run_paths.prior_features,
    )
    return summary
