"""Parse benchmark log files and extract summary metrics."""
from __future__ import annotations

import re
import sys

def parse_log(path: str) -> None:
    with open(path, "rb") as f:
        content = f.read().decode("utf-8", errors="replace")
    lines = content.split("\n")
    keywords = ["p50", "p95", "ms", "RESULT", "SUMMARY", "DONE", "completed",
                "vscode", "kubernetes", "latency", "mxbai", "jina", "bge",
                "BAAI", "mixedbread", "===", "Timing", "queries", "INFO",
                "recall", "R@", "MRR", "baseline", "beat", "winner", "WINNER",
                "candidate", "rerank", "model_id"]
    for line in lines:
        if any(k.lower() in line.lower() for k in keywords):
            clean = re.sub(r"Batches:.*?it/s\]", "", line).strip()
            if clean and "Batches" not in clean and len(clean) > 5:
                print(clean[:300].encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/cpu_latency.log"
    parse_log(path)
