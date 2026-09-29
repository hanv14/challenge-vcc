#!/usr/bin/env python
"""Does switching Phase 2's adapter off change the model? (DECISIONS.md D110)

    python scripts/adapter_check.py --config configs/server.yaml \
        --run-name server_12k --seed 2

Phase 2 trains its core and its LoRA adapter together and validates them
together. Until D110 every prediction, and every rehearsal arm, activated the
context adapter alone, so the adapter was loaded and never used. This scores
the saved Phase 2 checkpoint on Phase 2's own held-out validation twice, with
the adapter on and with it off, on the same cells and the same targets, and
prints the two side by side. The gap is what the submissions lost.

Use the seed the run was made with: it decides which targets are held out.
Trains nothing, writes `reports/adapter_check.json`. A few minutes on a GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()

from vccp.config import ConfigError, load_config  # noqa: E402
from vccp.data import reference  # noqa: E402
from vccp.logging_utils import get_logger, setup_logging  # noqa: E402
from vccp.models.checkpoint import load_core  # noqa: E402
from vccp.models.core import build_model  # noqa: E402
from vccp.paths import DataPaths, RunPaths  # noqa: E402
from vccp.phases import phase1, phase2  # noqa: E402
from vccp.priors.build import PriorFeatures  # noqa: E402
from vccp.runtime import resolve_device, seed_everything  # noqa: E402

#: The readings printed, in order. Each is a ratio against predicting no
#: change on the same cells, so lower is better on every one.
KEYS = (
    "pert_mse_ratio_to_no_change",
    "pert_train_mse_ratio_to_no_change",
    "map_control_ratio_to_no_change",
    "map_perturbed_ratio_to_no_change",
    "map_delta_ratio_to_no_change",
)


def check(cfg) -> dict:
    log = get_logger()
    run_paths = RunPaths(cfg)
    device = resolve_device(cfg.device)
    features = PriorFeatures.load(run_paths.prior_features)
    axis = reference.load_gene_names(DataPaths(cfg).gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}
    contexts = phase2.load_contexts(cfg, device, gene_index)

    model = build_model(cfg, features).to(device)
    for name in (phase1.ADAPTER, phase2.ADAPTER):
        model.add_adapter(name)
    load_core(
        run_paths.core_checkpoint(phase2.PHASE), model,
        layout_hash=features.layout_hash(), strict=False,
    )

    scores = {}
    for label, adapters in (("adapter_on", [phase2.ADAPTER]), ("adapter_off", [])):
        model.use_adapters(adapters)
        # `evaluate` reseeds its own draw, so both readings see the same cells.
        scores[label] = phase2.evaluate(model, contexts, cfg, device, gene_index)

    log.info("Phase 2's checkpoint, scored with its adapter on and off (D110):")
    log.info("  %-36s %12s %12s", "", "adapter on", "adapter off")
    for key in KEYS:
        on, off = scores["adapter_on"].get(key), scores["adapter_off"].get(key)
        if on is not None and off is not None:
            log.info("  %-36s %12.4f %12.4f", key, on, off)
    return {"checkpoint": str(run_paths.core_checkpoint(phase2.PHASE)), **scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.run_name:
        cfg = dataclasses.replace(cfg, run_name=args.run_name)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    run_paths = RunPaths(cfg)
    run_paths.ensure()
    setup_logging(run_paths.log)
    seed_everything(cfg.seed)

    report = check(cfg)
    out = run_paths.root / "reports" / "adapter_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    get_logger().info("-> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
