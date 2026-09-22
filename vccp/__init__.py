"""Virtual Cell Challenge 2026 — prototype pipeline.

The package is named `vccp`, not `vcc`, so that `python -m vccp` can never
resolve to the challenge's own `vcc` packaging tool (PLAN.md §8.3).

Heavy libraries (numpy, torch, anndata, polars) are imported inside functions
rather than at module import, because thread-pool sizes are fixed at import
and `vccp/__main__.py` has to cap them first.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
