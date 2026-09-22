"""Do the priors know any biology? (CLAUDE.md §4.1, checklist item 4)

Two checks, both of which the specification names:

1. **Neighbours.** Members of a protein complex — ribosome, proteasome, the
   mitochondrial respiratory chain — should be each other's nearest
   neighbours in the feature-role prior. Reported as an enrichment over
   what picking neighbours at random would give, so the number means the
   same thing whatever the family's size.

2. **Response correlation.** Genes with similar target-role embeddings
   should have correlated Replogle knockdown responses. Reported as the
   Spearman correlation between embedding similarity and response
   similarity over sampled target pairs.

Both are diagnostics, not gates: they are logged and written to
`priors/checks.json`, and a family the data does not contain is reported as
skipped with the reason rather than quietly dropped. On `mini_data` several
families are too small to say anything, which is itself worth seeing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config
from ..data import manifest as manifest_mod
from ..data import reference
from ..data import replogle as replogle_data
from ..logging_utils import get_logger
from ..paths import DataPaths
from .build import FEATURE, TARGET, PriorFeatures
from .hgnc_block import GROUP_SEPARATOR, _symbol_lookup
from .leakage import rehearsal_rebuild_plan

#: A family needs at least this many members present for "are they each
#: other's neighbours?" to be a question with an answer.
MIN_FAMILY_SIZE = 3


def _cosine_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms < 1e-12, 1.0, norms)


def matching_groups(cfg: Config, axis: list[str]) -> dict[str, tuple[str, list[int]]]:
    """HGNC gene groups matching a configured pattern -> (pattern, members).

    Checked per *group*, not per pattern. "Mitochondrial complex I" and
    "Mitochondrial complex IV" both match the same pattern but are different
    machines, and their subunits have no reason to be each other's
    neighbours — pooling them asks a question with no right answer.
    """
    paths = DataPaths(cfg)
    try:
        hgnc_path = paths.hgnc
    except FileNotFoundError:
        return {}

    frame = reference.load_hgnc(hgnc_path)
    lookup = _symbol_lookup(frame, axis)
    groups = frame["gene_group"].fillna("")
    patterns = [p.lower() for p in cfg.priors.check_families]

    found: dict[str, tuple[str, list[int]]] = {}
    for position, symbol in enumerate(axis):
        row = lookup.get(symbol)
        if row is None:
            continue
        for name in str(groups.iloc[row]).split(GROUP_SEPARATOR):
            name = name.strip()
            if not name:
                continue
            matched = next((p for p in patterns if p in name.lower()), None)
            if matched is None:
                continue
            found.setdefault(name, (matched, []))[1].append(position)
    return found


def neighbour_check(cfg: Config, features: PriorFeatures) -> dict:
    """Are complex members each other's nearest neighbours?"""
    axis = features.gene_symbols
    prior = features.priors[FEATURE]
    unit = _cosine_normalize(prior.astype(np.float32))
    k = cfg.priors.check_n_neighbors

    results = []
    groups = matching_groups(cfg, axis)
    if not groups:
        return {
            "description": "no HGNC gene group matched priors.check_families",
            "role": FEATURE,
            "patterns": list(cfg.priors.check_families),
            "families": [],
        }

    for family, (pattern, positions) in sorted(groups.items()):
        if len(positions) < MIN_FAMILY_SIZE:
            results.append(
                {
                    "family": family,
                    "pattern": pattern,
                    "status": "skipped",
                    "reason": f"only {len(positions)} member(s) present; "
                    f"need {MIN_FAMILY_SIZE}",
                    "n_members": len(positions),
                }
            )
            continue

        member_set = set(positions)
        # Similarity of each member against every gene, one member at a time
        # so this never forms an n_genes x n_genes matrix.
        hit_fractions = []
        for position in positions:
            similarity = unit @ unit[position]
            similarity[position] = -np.inf  # never its own neighbour
            neighbours = np.argpartition(-similarity, k)[:k]
            hit_fractions.append(len({int(n) for n in neighbours} & member_set) / k)

        observed = float(np.mean(hit_fractions))
        expected = (len(positions) - 1) / (len(axis) - 1)
        results.append(
            {
                "family": family,
                "pattern": pattern,
                "status": "ok",
                "n_members": len(positions),
                "n_neighbors": k,
                "observed_same_family_fraction": round(observed, 4),
                "expected_by_chance": round(expected, 6),
                "enrichment": round(observed / expected, 2) if expected > 0 else None,
                "enriched": bool(observed > expected),
            }
        )

    return {
        "description": "fraction of a complex member's nearest neighbours that are "
        "in the same complex, against the fraction chance would give",
        "role": FEATURE,
        "patterns": list(cfg.priors.check_families),
        "families": results,
    }


def response_correlation_check(cfg: Config, features: PriorFeatures) -> dict:
    """Do targets with similar target-role priors respond alike?

    The comparison is between two similarity structures — embedding cosine
    and response correlation — over the same pairs of targets, so it is
    reported as a rank correlation rather than as an absolute number.
    """
    paths = DataPaths(cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    prior = features.priors[TARGET]
    index = {symbol: i for i, symbol in enumerate(features.gene_symbols)}
    rng = np.random.default_rng(cfg.seed)

    per_context = []
    for context, directory in sorted(paths.discover_phase2_contexts().items()):
        ctx = replogle_data.load_replogle_context(directory)
        obs, matrix, _ = replogle_data.load_pseudobulk(directory, manifest.control_label)
        labels = obs.index.astype(str).tolist()
        control_row = labels.index(manifest.control_label)
        _, ctrl_std = ctx.control_stats()

        usable = [
            (row, label)
            for row, label in enumerate(labels)
            if label != manifest.control_label and label in index
        ]
        if len(usable) < MIN_FAMILY_SIZE:
            per_context.append(
                {
                    "context": context,
                    "status": "skipped",
                    "reason": f"only {len(usable)} target(s) are challenge genes",
                }
            )
            continue

        if len(usable) > cfg.priors.check_max_targets:
            chosen = rng.choice(len(usable), cfg.priors.check_max_targets, replace=False)
            usable = [usable[i] for i in sorted(chosen)]

        rows = [row for row, _ in usable]
        positions = [index[label] for _, label in usable]

        responses = (matrix[rows] - matrix[control_row]) / ctrl_std
        response_similarity = _pairwise_upper(np.corrcoef(responses))
        embedding_similarity = _pairwise_upper(
            _cosine_normalize(prior[positions]) @ _cosine_normalize(prior[positions]).T
        )

        finite = np.isfinite(response_similarity) & np.isfinite(embedding_similarity)
        if finite.sum() < 2:
            per_context.append(
                {"context": context, "status": "skipped", "reason": "no comparable pairs"}
            )
            continue

        from scipy.stats import spearmanr

        rho, p_value = spearmanr(
            embedding_similarity[finite], response_similarity[finite]
        )
        per_context.append(
            {
                "context": context,
                "status": "ok",
                "n_targets": len(usable),
                "n_pairs": int(finite.sum()),
                "spearman": round(float(rho), 4),
                "p_value": float(p_value),
                "positive": bool(rho > 0),
            }
        )

    return {
        "description": "rank correlation between target-role embedding similarity and "
        "Replogle knockdown-response correlation, over target pairs",
        "role": TARGET,
        "contexts": per_context,
    }


def magnitude_check(cfg: Config, features: PriorFeatures) -> dict:
    """How many knockdowns barely did anything.

    The phenotype blocks z-score each response before the direction
    decomposition, which turns a near-null response into a confident-looking
    random direction. The magnitude and quality scalars are how the model
    tells the two apart; this reports how much of the target set is in that
    position, per source, so the reader knows how much weight the direction
    features deserve.
    """
    rows = []
    for _, row in features.coverage.iterrows():
        if not row["built"] or not str(row["block"]).startswith("pert_phenotype"):
            continue
        detail = json.loads(row["detail"]) if row["detail"] else {}
        magnitude = detail.get("magnitude")
        if not magnitude:
            continue
        rows.append(
            {
                "block": str(row["block"]),
                "n_targets": detail.get("n_targets"),
                **magnitude,
                "quality_columns": detail.get("quality_columns", []),
                "n_imputed_quality_values": detail.get("n_imputed_quality_values", {}),
            }
        )

    return {
        "description": "fraction of targets whose knockdown response is below "
        "priors.phenotype_magnitude_floor x the source's median response magnitude; "
        "their direction features are noise, and the magnitude and quality scalars "
        "in the same block are what says so",
        "caveat": "where magnitude correlates with the source's depth signal, the "
        "fraction below the floor reflects how many cells a target had rather than "
        "how little its knockdown did; the depth signal is a feature of the same "
        "block so the model can discount it",
        "floor_fraction_of_median": cfg.priors.phenotype_magnitude_floor,
        "weighted_by_magnitude": cfg.priors.phenotype_weight_by_magnitude,
        "blocks": rows,
    }


def _pairwise_upper(square: np.ndarray) -> np.ndarray:
    """The strict upper triangle of a similarity matrix, flattened."""
    rows, cols = np.triu_indices(square.shape[0], k=1)
    return np.asarray(square)[rows, cols]


def run_checks(cfg: Config, features: PriorFeatures) -> dict:
    log = get_logger()

    neighbours = neighbour_check(cfg, features)
    responses = response_correlation_check(cfg, features)
    magnitudes = magnitude_check(cfg, features)
    leakage = rehearsal_rebuild_plan(
        features, list(DataPaths(cfg).discover_phase2_contexts())
    )

    log.info("prior checks — complex neighbours (feature role):")
    scored = [e for e in neighbours["families"] if e["status"] == "ok"]
    for entry in neighbours["families"]:
        if entry["status"] == "skipped":
            log.info("  %-52s skipped: %s", entry["family"][:52], entry["reason"])
        else:
            log.info(
                "  %-52s %2d members, %3.0f%% of top-%d in group (chance %.2f%%, %sx)",
                entry["family"][:52],
                entry["n_members"],
                100 * entry["observed_same_family_fraction"],
                entry["n_neighbors"],
                100 * entry["expected_by_chance"],
                entry["enrichment"],
            )
    if scored:
        log.info(
            "  %d of %d scored group(s) enriched over chance",
            sum(e["enriched"] for e in scored),
            len(scored),
        )

    log.info("prior checks — knockdown-response correlation (target role):")
    for entry in responses["contexts"]:
        if entry["status"] == "skipped":
            log.info("  %-12s skipped: %s", entry["context"], entry["reason"])
        else:
            log.info(
                "  %-12s spearman %+.3f over %d pairs from %d targets",
                entry["context"],
                entry["spearman"],
                entry["n_pairs"],
                entry["n_targets"],
            )

    log.info("prior checks — knockdown magnitude (target role):")
    for entry in magnitudes["blocks"]:
        depth = entry.get("magnitude_vs_depth_correlation")
        confound = (
            f", corr(magnitude, {entry.get('depth_signal')}) = {depth:+.2f}"
            if depth is not None
            else ""
        )
        log.info(
            "  %-34s %3d targets, %4.1f%% below the floor (%.3f = %.2f x median %.3f)%s%s",
            entry["block"],
            entry["n_targets"] or 0,
            100 * entry["fraction_below_floor"],
            entry["floor"],
            entry["floor_fraction_of_median"],
            entry["median_rms"],
            confound,
            ", basis weighted" if entry["weighted_by_magnitude"] else "",
        )

    log.info("prior checks — rehearsal rebuild plan (leakage, §4.1):")
    for name, plan in sorted(leakage["variants"].items()):
        log.info(
            "  %-28s %d rebuilt, %d dropped, %d reused",
            name,
            plan["n_rebuilt"],
            plan["n_dropped"],
            plan["n_reused"],
        )

    return {
        "neighbour_check": neighbours,
        "response_correlation_check": responses,
        "magnitude_check": magnitudes,
        "rehearsal_rebuild_plan": leakage,
    }


def summarize(checks: dict) -> pd.DataFrame:
    """A flat table of the checks, for the results summary at M5."""
    rows = []
    for entry in checks["neighbour_check"]["families"]:
        rows.append({"check": "neighbours", "subject": entry["family"], **entry})
    for entry in checks["response_correlation_check"]["contexts"]:
        rows.append({"check": "response_correlation", "subject": entry["context"], **entry})
    for entry in checks.get("magnitude_check", {}).get("blocks", []):
        rows.append({"check": "magnitude", "subject": entry["block"], **entry})
    return pd.DataFrame(rows)
