from __future__ import annotations

import collections
import glob
import hashlib
import json
import sys
import warnings

import joblib
import lightgbm
import numpy
import pandas
import sklearn
from sklearn.base import BaseEstimator

REC: list[tuple[str, str]] = []
_orig = BaseEstimator.__setstate__


def _rec(self, state):  # type: ignore[no-untyped-def]
    REC.append((type(self).__name__, str(state.get("_sklearn_version", "pre-0.18"))))
    return _orig(self, state)


BaseEstimator.__setstate__ = _rec  # type: ignore[method-assign]
print("RUNTIME sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__,
      "numpy", numpy.__version__, "pandas", pandas.__version__, "joblib", joblib.__version__)
root = sys.argv[1]
files = sorted(glob.glob(root + "/*.pkl") + glob.glob(root + "/*/meta.pkl"))
for f in files:
    REC.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = joblib.load(f)
    sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
    c = collections.Counter(REC)
    vers = sorted({v for _, v in REC})
    print(f"\n{f.split('/models/')[-1]} sha256={sha[:16]} stored_sklearn_versions={vers} "
          f"n_estimators={len(REC)} warnings={len(w)}")
    print("  top-level keys:", sorted(obj.keys()) if isinstance(obj, dict) else type(obj))
    print("  estimators:", dict(sorted((f"{k[0]}@{k[1]}", n) for k, n in c.items())))
    if isinstance(obj, dict):
        for k in ("model_point", "model_q10", "model_q90", "model_bucket"):
            m = obj.get(k)
            if m is not None:
                try:
                    s = m.model_to_string()[:200] if hasattr(m, "model_to_string") else repr(type(m))
                    print(f"  {k}: {type(m).__module__}.{type(m).__name__} head={s.splitlines()[:2]}")
                except Exception as e:  # noqa: BLE001
                    print(f"  {k}: {type(m)} err {e}")
        for k, v in obj.items():
            if hasattr(v, "__module__") and "pandas" in str(type(v)):
                print(f"  {k}: pandas obj {type(v)}")
