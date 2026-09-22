"""The gene vocabulary: prior blocks, their assembly, and their checks.

CLAUDE.md §4.1. Each block turns one data source into gene features; the
blocks are reduced, standardized and concatenated into one prior per role,
which `vccp/models/embedding.py` projects and offsets.
"""
