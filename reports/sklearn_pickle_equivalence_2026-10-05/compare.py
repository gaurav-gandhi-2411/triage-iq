from __future__ import annotations

import sys

import numpy as np

a = np.load(sys.argv[1], allow_pickle=True)
b = np.load(sys.argv[2], allow_pickle=True)
assert set(a.files) == set(b.files)
print(f"{'array':62s} {'shape':>12s} {'exact_eq':>10s} {'max_abs_diff':>14s}")
for k in sorted(a.files):
    x, y = a[k], b[k]
    if x.dtype.kind in "US" or x.dtype == object:
        n = int((x == y).sum())
        print(f"{k:62s} {str(x.shape):>12s} {n}/{x.size:<7d} {'(labels)':>14s}")
    else:
        eq = int(np.array_equal(x, y)) if x.ndim == 0 else int((x == y).all(axis=tuple(range(1, x.ndim))).sum()) if x.ndim > 1 else int((x == y).sum())
        rows = x.shape[0] if x.ndim else 1
        print(f"{k:62s} {str(x.shape):>12s} {eq}/{rows:<7d} {float(np.max(np.abs(x - y))) if x.size else 0.0:14.3e}")
# top-3 sets for classifier outputs
print("\nclassifier top-3 label-set / top-1 agreement:")
for k in sorted(a.files):
    if k.endswith("|cal"):
        t1 = (a[k].argmax(1) == b[k].argmax(1)).mean()
        s = np.argsort(-a[k], 1)[:, :3]
        s2 = np.argsort(-b[k], 1)[:, :3]
        ord_eq = (s == s2).all(1).mean()
        print(f"{k:50s} n={len(a[k])} top1_same={t1:.6f} top3_ordered_same={ord_eq:.6f}")
