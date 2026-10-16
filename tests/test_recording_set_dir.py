"""TRIAGEIQ_RECORDING_SET_DIR points the recorder at a separate set; unset, every path is unchanged.

A fresh interpreter per case: the paths are module-level constants resolved at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SNIPPET = (
    "import json, sys; sys.path[:0] = [{src!r}, {ev!r}]; import record_cassettes as rc; "
    "print(json.dumps([str(p) for p in (rc.EVAL_SET, rc.CASSETTE_PATH, rc.CHECKPOINT_PATH, rc.STATUS_PATH)]))"
).format(src=str(ROOT / "src"), ev=str(ROOT / "eval"))


def _paths(set_dir: Path | None) -> list[Path]:
    env = {k: v for k, v in os.environ.items() if k != "TRIAGEIQ_RECORDING_SET_DIR"}
    if set_dir is not None:
        env["TRIAGEIQ_RECORDING_SET_DIR"] = str(set_dir)
    out = subprocess.run([sys.executable, "-c", _SNIPPET], env=env, capture_output=True, text=True, check=True)  # noqa: S603
    return [Path(p) for p in json.loads(out.stdout.strip().splitlines()[-1])]


def test_unset_keeps_the_committed_paths(tmp_path: Path) -> None:
    eval_set, cassette, checkpoint, status = _paths(None)
    assert eval_set == ROOT / "eval" / "eval_set.jsonl"
    assert cassette == ROOT / "eval" / "cassettes" / "eval_cassette.json"
    assert checkpoint == ROOT / "eval" / "cassettes" / "recording_checkpoint.json"
    assert status == ROOT / "eval" / "cassettes" / "RECORDING_STATUS.txt"


def test_set_dir_moves_every_recorder_path_and_none_stay_in_the_committed_cassette_dir(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert paths[0] == tmp_path / "eval_set.jsonl"
    assert all(p.parent == tmp_path / "cassettes" for p in paths[1:])
    assert not any(ROOT / "eval" / "cassettes" in p.parents for p in paths)
