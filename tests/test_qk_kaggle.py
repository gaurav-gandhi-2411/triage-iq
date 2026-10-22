"""Offline tests for scripts/qk_kaggle.py (subprocess mocked, hermetic tmp_path roots)."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import qk_kaggle as qk  # noqa: E402


class Recorder:
    """Stand-in for subprocess.run recording command lines."""

    def __init__(self, stdout: str = "", rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.stdout, self.rc = stdout, rc

    def __call__(self, cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.rc, self.stdout, "boom")


def _make_data(root: Path) -> None:
    d = root / "derived" / "o__r"
    d.mkdir(parents=True)
    pd.DataFrame({"number": [1, 2], "title": ["t1", "t2"], "body": ["b" * 5000, None]}).to_parquet(
        d / "issues.parquet"
    )
    pd.DataFrame(
        {
            "number": [10, 11, 12],
            "title": ["p10", "p11", "p12"],
            "body": ["x", "y", "z"],
            "merged": [True, False, True],
        }
    ).to_parquet(d / "prs.parquet")
    pf = root / "raw" / "o__r" / "pr_files"
    pf.mkdir(parents=True)
    rec = {"number": 10, "files": {"nodes": [{"path": "a/b.go"}, {"path": "c.md"}]}}
    with gzip.open(pf / "0.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_build_texts_issues_truncates_body(tmp_path: Path) -> None:
    _make_data(tmp_path)
    df = qk.build_texts("o/r", "issues", tmp_path)
    assert list(df.columns) == ["repo", "kind", "number", "text"]
    assert df.text[0] == "t1\n" + "b" * 4000
    assert df.text[1] == "t2\n"  # null body tolerated


def test_build_texts_prs_merged_only_with_files(tmp_path: Path) -> None:
    _make_data(tmp_path)
    df = qk.build_texts("o/r", "prs", tmp_path)
    assert df.number.tolist() == [10, 12]
    assert df.text[0] == "p10\nx\na/b.go\nc.md"
    assert df.text[1] == "p12\nz\n"  # no file record -> empty path list


@pytest.mark.parametrize("exists", [False, True])
def test_build_dataset_private_create_or_version(tmp_path: Path, monkeypatch, exists) -> None:
    data, stage = tmp_path / "data", tmp_path / "stage"
    _make_data(data)
    ref = "me/triageiq-texts-o-r-issues"
    rec = Recorder(stdout=ref if exists else "")
    monkeypatch.setattr(subprocess, "run", rec)
    qk.main(
        [
            "--owner",
            "me",
            "--staging",
            str(stage),
            "--kaggle-exe",
            "kg",
            "build-dataset",
            "--repo",
            "o/r",
            "--kind",
            "issues",
            "--data-root",
            str(data),
        ]
    )
    meta = json.loads((stage / "triageiq-texts-o-r-issues" / "dataset-metadata.json").read_text())
    assert meta["isPrivate"] is True and meta["id"] == ref
    last = rec.calls[-1]
    assert last[1:3] == ["datasets", "version" if exists else "create"]
    assert "-t" in last and "--public" not in last and "-u" not in last
    assert (stage / "triageiq-texts-o-r-issues" / "o__r__issues.parquet").exists()


def test_push_kernel_private_gpu_and_no_credentials(tmp_path: Path, monkeypatch, capsys) -> None:
    src = tmp_path / "k"
    src.mkdir()
    (src / "embed_bge.py").write_text("print(1)")
    (src / "kernel-metadata.json").write_text("stale")
    rec = Recorder(stdout="Kernel version 1 successfully pushed.")
    monkeypatch.setattr(subprocess, "run", rec)
    qk.main(
        [
            "--owner",
            "me",
            "--staging",
            str(tmp_path / "s"),
            "--kaggle-exe",
            "kg",
            "push-kernel",
            "--kernel",
            str(src),
            "--slug",
            "kk",
            "--datasets",
            "me/a,me/b",
        ]
    )
    meta = json.loads((tmp_path / "s" / "kernel-kk" / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == ["me/a", "me/b"]
    printed = capsys.readouterr().out + " ".join(" ".join(c) for c in rec.calls)
    assert "kaggle.json" not in printed and "key" not in printed.lower()


def test_failure_raises_with_stderr(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", Recorder(rc=1))
    with pytest.raises(RuntimeError, match="boom"):
        qk.kaggle(Path("kg"), ["kernels", "status", "x"])


def test_status_parse_and_wait_exit_codes(monkeypatch) -> None:
    seq = iter(["running", "KernelWorkerStatus.COMPLETE"])
    monkeypatch.setattr(
        subprocess, "run", lambda c, **_: subprocess.CompletedProcess(c, 0, next(seq), "")
    )
    monkeypatch.setattr(qk.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit) as e:
        qk.main(["--kaggle-exe", "kg", "wait", "--kernel", "k", "--timeout", "100"])
    assert e.value.code == 0


def test_wait_error_and_timeout(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="KernelWorkerStatus.ERROR"))
    with pytest.raises(SystemExit) as e:
        qk.main(["--kaggle-exe", "kg", "wait", "--kernel", "k"])
    assert e.value.code == 1
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="running"))
    monkeypatch.setattr(qk.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit) as e:
        qk.main(["--kaggle-exe", "kg", "wait", "--kernel", "k", "--timeout", "0"])
    assert e.value.code == 2


def test_download_creates_out_dir(tmp_path: Path, monkeypatch) -> None:
    rec = Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    out = tmp_path / "deep" / "slug"
    qk.main(["--kaggle-exe", "kg", "--owner", "me", "download", "--kernel", "k", "--out", str(out)])
    assert out.is_dir()
    assert rec.calls[0][1:] == ["kernels", "output", "me/k", "-p", str(out)]


def test_push_kernel_exit0_error_is_not_success(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "k"
    src.mkdir()
    (src / "embed_bge.py").write_text("print(1)")
    monkeypatch.setattr(
        subprocess, "run", Recorder(stdout="Kernel push error: Maximum batch GPU session count")
    )
    argv = ["--staging", str(tmp_path / "s"), "--kaggle-exe", "kg", "push-kernel"]
    with pytest.raises(RuntimeError, match="not confirmed"):
        qk.main([*argv, "--kernel", str(src), "--slug", "kk", "--datasets", "me/a"])


def test_push_kernel_text_prefix_written_exactly(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "k"
    src.mkdir()
    (src / "embed_bge.py").write_text("print(1)")
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="Kernel version 2 successfully pushed."))
    prefix = "Represent this sentence for searching relevant passages: "
    argv = ["--owner", "me", "--staging", str(tmp_path / "s"), "--kaggle-exe", "kg", "push-kernel"]
    argv += ["--kernel", str(src), "--slug", "kq", "--datasets", "me/a"]
    qk.main([*argv, "--text-prefix", prefix])
    assert (tmp_path / "s" / "kernel-kq" / "text_prefix.txt").read_text() == prefix
    qk.main(argv)  # no prefix -> the restaged dir carries no prefix file
    assert not (tmp_path / "s" / "kernel-kq" / "text_prefix.txt").exists()


def test_kernel_load_prefix(tmp_path: Path) -> None:
    kp = Path(__file__).resolve().parents[1] / "kaggle" / "kernels" / "embed_bge" / "embed_bge.py"
    src = kp.read_text(encoding="utf-8")
    fn_src = src[src.index("def load_prefix") : src.index("def encode")]
    ns: dict = {"Path": Path, "PREFIX_FILE": tmp_path / "none.txt"}
    exec(fn_src, ns)  # noqa: S102 - the kernel imports torch; exec only the pure helper
    assert ns["load_prefix"]() == ""
    (tmp_path / "p.txt").write_text("Q: ", encoding="utf-8")
    assert ns["load_prefix"](tmp_path / "p.txt") == "Q: "
