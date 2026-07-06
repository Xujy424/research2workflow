"""Small runnable demo for MatrixStore.

This script uses a temporary data root, so it will not touch ``D:/data``.
Run from the repository root:

    python v2/ResearchFlow/examples/matrix_store_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ResearchFlow.matrix_store import MatrixStore


def build_demo_store() -> tuple[MatrixStore, Path]:
    root = Path(tempfile.mkdtemp(prefix="matrix_store_demo_"))
    axis_dir = root / "axis"
    axis_dir.mkdir(parents=True)

    np.save(axis_dir / "date.npy", np.array(["2026-07-03", "2026-07-06", "2026-07-07"]))
    np.save(axis_dir / "tick.npy", np.array(["000001.SZ", "000002.SZ", "600000.SH", "300750.SZ"]))
    return MatrixStore(root), root


def main() -> None:
    store, root = build_demo_store()
    axis = store.load_axis()
    print(f"demo data root: {root}")
    print(f"axis shape: {axis.shape}")

    store.ensure_matrix("factorpool", "my_factor")

    # Full rewrite: historical backfill or full recomputation.
    full_matrix = np.arange(np.prod(axis.shape), dtype=float).reshape(axis.shape)
    store.write_matrix("factorpool", "my_factor", full_matrix)

    # Row update: one date, all stocks.
    store.update_slice(
        "factorpool",
        "my_factor",
        np.array([1.1, 1.2, 1.3, 1.4]),
        dates="2026-07-06",
    )

    # Column update: one stock, all dates.
    store.update_slice(
        "factorpool",
        "my_factor",
        np.array([10.0, 20.0, 30.0]),
        ticks="000002.SZ",
    )

    # Block update: multiple dates x multiple stocks.
    store.update_slice(
        "factorpool",
        "my_factor",
        np.array([[5.1, 5.2], [6.1, 6.2]]),
        dates=["2026-07-03", "2026-07-07"],
        ticks=["000001.SZ", "600000.SH"],
    )

    # Paired-cell update: (date_i, tick_i) one-to-one.
    store.update_slice(
        "factorpool",
        "my_factor",
        np.array([99.0, 88.0]),
        dates=["2026-07-03", "2026-07-07"],
        ticks=["300750.SZ", "000002.SZ"],
        paired=True,
    )

    print("full matrix:")
    print(store.read_slice("factorpool", "my_factor"))

    print("one row:")
    print(store.read_slice("factorpool", "my_factor", dates="2026-07-06"))

    print("one column:")
    print(store.read_slice("factorpool", "my_factor", ticks="000002.SZ"))

    print("paired cells:")
    print(
        store.read_slice(
            "factorpool",
            "my_factor",
            dates=["2026-07-03", "2026-07-07"],
            ticks=["300750.SZ", "000002.SZ"],
            paired=True,
        )
    )


if __name__ == "__main__":
    main()
