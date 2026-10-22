"""Kaggle embedding pipeline driver (free GPU): build private datasets, push kernel, wait, download.

Subcommands: build-dataset, push-kernel, status, wait, download. Every Kaggle call goes through
the `kaggle` CLI installed in a dedicated venv (path configurable); credentials are read by the CLI
from its default location and are never touched or logged here. Datasets/kernels are always private.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(r"D:\triageiq-data")
DEFAULT_KAGGLE = DATA_ROOT / "venvs" / "kaggle" / "Scripts" / "kaggle.exe"
STAGING = DATA_ROOT / "kaggle_staging"
OWNER = "gauravgandhi2411"  # public Kaggle username, not a credential
ISSUE_BODY_CHARS, PR_BODY_CHARS, PR_MAX_FILES = 4000, 2000, 100
TERMINAL = {"complete": True, "error": False, "cancel": False}


def slug_of(repo: str) -> str:
    """'owner/repo' -> 'owner__repo' (the on-disk slug)."""
    return repo.replace("/", "__")


def kaggle(exe: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the kaggle CLI; raise with its stderr on failure."""
    cmd = [str(exe), *args]
    print("+", " ".join(args), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if res.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])} failed: {res.stderr.strip()[-800:]}")
    return res


def _pr_files(raw_dir: Path) -> dict[int, list[str]]:
    """Map PR number -> changed file paths from raw pr_files/*.jsonl.gz."""
    out: dict[int, list[str]] = {}
    for p in sorted((raw_dir / "pr_files").glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["number"]] = [n["path"] for n in rec["files"]["nodes"]]
    return out


def build_texts(repo: str, kind: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Return DataFrame(repo, kind, number, text) for issues or merged PRs."""
    slug = slug_of(repo)
    v2 = data_root / "derived_v2" / slug / f"{kind}.parquet"
    src = v2 if v2.exists() else data_root / "derived" / slug / f"{kind}.parquet"
    df = pd.read_parquet(src)
    title, body = df["title"].fillna(""), df["body"].fillna("")
    if kind == "issues":
        text = title + "\n" + body.str.slice(0, ISSUE_BODY_CHARS)
    else:
        df = df[df["merged"].astype(bool)].reset_index(drop=True)
        files = _pr_files(data_root / "raw" / slug)
        title, body = df["title"].fillna(""), df["body"].fillna("")
        paths = df["number"].map(lambda n: "\n".join(files.get(int(n), [])[:PR_MAX_FILES]))
        text = title + "\n" + body.str.slice(0, PR_BODY_CHARS) + "\n" + paths
    return pd.DataFrame(
        {"repo": repo, "kind": kind, "number": df["number"].astype("int64"), "text": text}
    )


def cmd_build_dataset(a: argparse.Namespace) -> None:
    """Write parquet + metadata to staging, then create (or version) a PRIVATE dataset."""
    slug = slug_of(a.repo)
    ds_slug = f"triageiq-texts-{slug.replace('__', '-').replace('_', '-')}-{a.kind}"
    out = Path(a.staging) / ds_slug
    out.mkdir(parents=True, exist_ok=True)
    df = build_texts(a.repo, a.kind, Path(a.data_root))
    df.to_parquet(out / f"{slug}__{a.kind}.parquet", index=False)
    meta = {
        "title": ds_slug,
        "id": f"{a.owner}/{ds_slug}",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{ds_slug}: {len(df)} rows -> {out}")
    exe = Path(a.kaggle_exe)
    listing = kaggle(exe, ["datasets", "list", "--mine", "--search", ds_slug, "--csv"]).stdout
    # -t keeps parquet as parquet (default converts tabular files to CSV); no --public ever.
    if meta["id"] in listing:
        kaggle(exe, ["datasets", "version", "-p", str(out), "-m", "refresh", "-t"])
    else:
        kaggle(exe, ["datasets", "create", "-p", str(out), "-t"])


def kernel_metadata(owner: str, slug: str, datasets: list[str]) -> dict:
    """Private GPU script-kernel metadata with internet on (HF model download, pinned revision)."""
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "embed_bge.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def cmd_push_kernel(a: argparse.Namespace) -> None:
    """Copy the kernel dir to staging, write metadata there, push (starts the run)."""
    stage = Path(a.staging) / f"kernel-{a.slug}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(a.kernel, stage, ignore=shutil.ignore_patterns("kernel-metadata.json"))
    meta = kernel_metadata(a.owner, a.slug, [d for d in a.datasets.split(",") if d])
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    args = ["kernels", "push", "-p", str(stage), "--accelerator", a.accelerator]
    res = kaggle(Path(a.kaggle_exe), args)
    text = (res.stdout + res.stderr).strip()
    # The CLI exits 0 on "Maximum batch GPU session count reached" etc.; a real push prints
    # a "Kernel version N successfully pushed" line, so require it.
    if "successfully pushed" not in text.lower():
        raise RuntimeError(f"kernel push not confirmed: {text[-400:]}")
    print(text)


def kernel_status(exe: Path, owner: str, slug: str) -> str:
    """Return lowercase status token: queued/running/complete/error/cancel*."""
    out = kaggle(exe, ["kernels", "status", f"{owner}/{slug}"]).stdout.lower()
    for token in ("complete", "error", "cancel", "running", "queued"):
        if token in out:
            return token
    return "unknown"


def cmd_status(a: argparse.Namespace) -> None:
    print(kernel_status(Path(a.kaggle_exe), a.owner, a.kernel))


def cmd_wait(a: argparse.Namespace) -> None:
    """Poll every `interval` s up to `timeout` s; exit 0 only on complete."""
    deadline = time.monotonic() + a.timeout
    while True:
        st = kernel_status(Path(a.kaggle_exe), a.owner, a.kernel)
        print(time.strftime("%H:%M:%S"), st, flush=True)
        if st in TERMINAL:
            sys.exit(0 if TERMINAL[st] else 1)
        if time.monotonic() >= deadline:
            print("timeout")
            sys.exit(2)
        time.sleep(a.interval)


def cmd_download(a: argparse.Namespace) -> None:
    """Download kernel output files into --out (created if missing)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kaggle(Path(a.kaggle_exe), ["kernels", "output", f"{a.owner}/{a.kernel}", "-p", str(out)])
    print("downloaded:", sorted(p.name for p in out.iterdir()))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kaggle-exe", default=str(DEFAULT_KAGGLE))
    p.add_argument("--owner", default=OWNER)
    p.add_argument("--staging", default=str(STAGING))
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-dataset")
    b.add_argument("--repo", required=True)
    b.add_argument("--kind", required=True, choices=["issues", "prs"])
    b.add_argument("--data-root", default=str(DATA_ROOT))
    b.set_defaults(fn=cmd_build_dataset)
    k = sub.add_parser("push-kernel")
    k.add_argument("--kernel", required=True, help="kernel source dir (contains embed_bge.py)")
    k.add_argument("--slug", required=True, help="kernel slug to create/update")
    k.add_argument("--datasets", required=True, help="comma-separated owner/dataset refs")
    k.add_argument("--accelerator", default="NvidiaTeslaT4")
    k.set_defaults(fn=cmd_push_kernel)
    s = sub.add_parser("status")
    s.add_argument("--kernel", required=True)
    s.set_defaults(fn=cmd_status)
    w = sub.add_parser("wait")
    w.add_argument("--kernel", required=True)
    w.add_argument("--timeout", type=int, default=7200)
    w.add_argument("--interval", type=int, default=60)
    w.set_defaults(fn=cmd_wait)
    d = sub.add_parser("download")
    d.add_argument("--kernel", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_download)
    return p


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
