"""Can this `.h5ad` actually be read? (CLAUDE.md §1.1)

`check-data` verifies that every file, column and gene order is what the
pipeline expects — but it reads *metadata*. A file whose `obs` and `var` are
perfect can still fail hours later, in the middle of the priors, with

    OSError: Can't synchronously read data (filter returned failure during read)

which is HDF5 saying a compressed chunk could not be decoded: the file was
truncated or damaged in transit, or it was written with a compression filter
this environment has no plugin for.

So this module *touches the matrices*. It reads blocks of each `X` and each
layer, spread across the file, and reports the first element that fails with
the path, the element, the row range and the filter in use. Small datasets are
read in full; a 65 GB single-cell file is sampled, so a pass here means
"nothing broken was found", not "every byte is sound".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Read every byte of an element up to this size; sample above it.
FULL_READ_BYTES = 2 * 1024**3
#: Row blocks sampled from an element too large to read in full.
DEFAULT_SAMPLES = 12
DEFAULT_BLOCK_ROWS = 512


@dataclass
class ElementProbe:
    """One matrix element of one file."""

    element: str
    ok: bool
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    encoding: str | None = None
    filters: list[str] = field(default_factory=list)
    n_rows_read: int = 0
    full_read: bool = False
    error: str | None = None
    failed_rows: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "ok": self.ok,
            "shape": list(self.shape) if self.shape else None,
            "dtype": self.dtype,
            "encoding": self.encoding,
            "filters": self.filters,
            "rows_read": self.n_rows_read,
            "full_read": self.full_read,
            "error": self.error,
            "failed_rows": list(self.failed_rows) if self.failed_rows else None,
        }


@dataclass
class FileProbe:
    path: str
    ok: bool
    elements: list[ElementProbe] = field(default_factory=list)
    error: str | None = None

    @property
    def failures(self) -> list[ElementProbe]:
        return [e for e in self.elements if not e.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "error": self.error,
            "elements": [e.as_dict() for e in self.elements],
        }

    def summary(self) -> str:
        if self.error:
            return f"[FAIL] {self.path}: {self.error}"
        if self.ok:
            read = sum(e.n_rows_read for e in self.elements)
            how = "read in full" if all(e.full_read for e in self.elements) else "sampled"
            return (
                f"[ok  ] {self.path}: {len(self.elements)} element(s), "
                f"{read:,} rows {how}"
            )
        lines = [f"[FAIL] {self.path}"]
        for element in self.failures:
            where = (
                f" at rows {element.failed_rows[0]}–{element.failed_rows[1]}"
                if element.failed_rows else ""
            )
            filters = f", filters {element.filters}" if element.filters else ""
            lines.append(f"         {element.element}{where}{filters}: {element.error}")
        return "\n".join(lines)


def _filters(dataset) -> list[str]:
    """The compression filters a dataset was written with."""
    names = []
    try:
        plist = dataset.id.get_create_plist()
        for index in range(plist.get_nfilters()):
            code, _flags, _values, name = plist.get_filter(index)[:4]
            label = name.decode() if isinstance(name, bytes) else str(name)
            names.append(label or f"filter-{code}")
    except Exception:  # noqa: BLE001 — filter reporting must not fail the probe
        if getattr(dataset, "compression", None):
            names.append(str(dataset.compression))
    return names


def _blocks(n_rows: int, samples: int, block_rows: int) -> list[tuple[int, int]]:
    """Row ranges to read: the start, the end, and evenly spaced in between."""
    if n_rows <= samples * block_rows:
        return [(0, n_rows)]
    starts = np.unique(
        np.linspace(0, n_rows - block_rows, samples).astype(np.int64)
    )
    return [(int(start), int(min(start + block_rows, n_rows))) for start in starts]


def _probe_sparse(group, name: str, samples: int, block_rows: int) -> ElementProbe:
    import h5py

    encoding = group.attrs.get("encoding-type", "")
    shape = tuple(int(v) for v in group.attrs.get("shape", ()))
    data = group["data"]
    probe = ElementProbe(
        element=name, ok=True, shape=shape, dtype=str(data.dtype),
        encoding=str(encoding), filters=_filters(data),
    )

    # `indptr` is one entry per row and is what says where a row's values sit.
    indptr = group["indptr"][:]
    n_rows = len(indptr) - 1
    nnz = int(data.shape[0])
    probe.full_read = nnz * data.dtype.itemsize <= FULL_READ_BYTES
    ranges = [(0, n_rows)] if probe.full_read else _blocks(n_rows, samples, block_rows)

    indices = group["indices"]
    for start, stop in ranges:
        lo, hi = int(indptr[start]), int(indptr[stop])
        try:
            data[lo:hi]
            indices[lo:hi]
        except (OSError, RuntimeError) as exc:
            probe.ok = False
            probe.error = str(exc).strip().splitlines()[0]
            probe.failed_rows = (start, stop)
            return probe
        probe.n_rows_read += stop - start
    return probe


def _probe_dense(dataset, name: str, samples: int, block_rows: int) -> ElementProbe:
    probe = ElementProbe(
        element=name, ok=True, shape=tuple(int(v) for v in dataset.shape),
        dtype=str(dataset.dtype), encoding="array", filters=_filters(dataset),
    )
    n_rows = int(dataset.shape[0]) if dataset.shape else 0
    probe.full_read = dataset.size * dataset.dtype.itemsize <= FULL_READ_BYTES
    ranges = [(0, n_rows)] if probe.full_read else _blocks(n_rows, samples, block_rows)

    for start, stop in ranges:
        try:
            dataset[start:stop]
        except (OSError, RuntimeError) as exc:
            probe.ok = False
            probe.error = str(exc).strip().splitlines()[0]
            probe.failed_rows = (start, stop)
            return probe
        probe.n_rows_read += stop - start
    return probe


def probe_h5ad(
    path: Path, *, samples: int = DEFAULT_SAMPLES, block_rows: int = DEFAULT_BLOCK_ROWS
) -> FileProbe:
    """Read blocks of every matrix in one `.h5ad`, and say what failed."""
    import h5py

    path = Path(path)
    if not path.is_file():
        return FileProbe(str(path), ok=False, error="file does not exist")

    try:
        handle = h5py.File(path, "r")
    except (OSError, RuntimeError) as exc:
        return FileProbe(str(path), ok=False, error=f"cannot be opened: {exc}")

    result = FileProbe(str(path), ok=True)
    try:
        elements: list[tuple[str, Any]] = []
        if "X" in handle:
            elements.append(("X", handle["X"]))
        for group in ("layers", "obsm", "varm"):
            if group in handle:
                elements.extend(
                    (f"{group}/{key}", handle[group][key]) for key in handle[group]
                )

        for name, element in elements:
            if isinstance(element, h5py.Group):
                if "data" in element and "indptr" in element:
                    probe = _probe_sparse(element, name, samples, block_rows)
                else:
                    continue
            else:
                probe = _probe_dense(element, name, samples, block_rows)
            result.elements.append(probe)
            if not probe.ok:
                result.ok = False
    finally:
        handle.close()
    return result


def probe_all(paths: list[Path], **kwargs) -> list[FileProbe]:
    return [probe_h5ad(path, **kwargs) for path in paths]


def explain(probe: FileProbe) -> str:
    """What a failure usually means, and what to do about it."""
    if probe.ok:
        return ""
    text = str(probe.error or "") + " ".join(e.error or "" for e in probe.elements)
    if "not registered" in text or "not available" in text:
        return (
            "HDF5 has no plugin for the compression this file was written with. "
            "`pip install hdf5plugin` and it will be found, or rewrite the file "
            "with gzip."
        )
    if "filter returned failure" in text:
        return (
            "A compressed chunk could not be decoded. Almost always the file is "
            "truncated or damaged — check its size against the source, compare "
            "checksums, and copy it again. It is not a bug in the reader: the "
            "same file fails for h5py, anndata and scanpy alike."
        )
    return "The file could not be read; the error above is HDF5's own."
