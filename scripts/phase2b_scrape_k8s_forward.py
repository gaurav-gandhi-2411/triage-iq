"""Phase 2b: k8s forward-scrape — issue/PR records #15,003-30,000 (GG decision (b)).

Uses GraphQL `issueOrPullRequest(number:)` batched 100 numbers per query (~150 queries,
minutes, well under the 5,000-points/hr GraphQL budget). A first attempt used REST
/issues offset pagination and failed: GitHub now 422s past page 100 (10,000-result offset
cap), and pages 1-99 are all inside the already-scraped #1-15,002 prefix — so REST listing
cannot reach the forward window at all.

Records are saved REST-shaped so `triage_iq.data.preprocess._extract_fields` reads them
unchanged (id, number, title, body, labels[{name}], state, created_at, closed_at,
user{login}, comments, assignees). PRs carry the standard `"pull_request": {}` marker.
Comments are NOT fetched: the k8s pair channel is body references ("fixes #N"), not
comments — each record carries "comments_skipped": true so provenance is explicit.

Output: data/raw/kubernetes_kubernetes/{n}.json for new numbers (existing files never touched)
Reproduce/resume: python scripts/phase2b_scrape_k8s_forward.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

GRAPHQL = "https://api.github.com/graphql"
RAW_DIR = Path("data/raw/kubernetes_kubernetes")
START, END = 15_003, 30_000  # forward-scrape window (ADR-0026 scope, GG-approved)
BATCH = 100

FRAGMENT = """
  __typename
  ... on Issue { databaseId number title body state createdAt closedAt
    author { login } comments { totalCount }
    labels(first: 30) { nodes { name } } assignees(first: 10) { nodes { login } } }
  ... on PullRequest { databaseId number title body state createdAt closedAt
    author { login } comments { totalCount }
    labels(first: 30) { nodes { name } } assignees(first: 10) { nodes { login } } }
"""


def to_rest_shape(node: dict) -> dict:
    state = str(node.get("state", "")).lower()
    rec = {
        "id": node.get("databaseId"),
        "number": node["number"],
        "title": node.get("title"),
        "body": node.get("body"),
        "state": "closed" if state == "merged" else state,
        "created_at": node.get("createdAt"),
        "closed_at": node.get("closedAt"),
        "user": {"login": (node.get("author") or {}).get("login")},
        "comments": (node.get("comments") or {}).get("totalCount", 0),
        "labels": [{"name": lb["name"]} for lb in (node.get("labels") or {}).get("nodes", [])],
        "assignees": [
            {"login": a["login"]} for a in (node.get("assignees") or {}).get("nodes", [])
        ],
        "comments_data": [],
        "comments_skipped": True,  # body-channel scrape; comments not fetched
    }
    if node.get("__typename") == "PullRequest":
        rec["pull_request"] = {}
    return rec


def main() -> None:
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.critical("GITHUB_TOKEN not set")
        sys.exit(1)
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    todo = [n for n in range(START, END + 1) if not (RAW_DIR / f"{n}.json").exists()]
    log.info("forward window %d-%d: %d numbers to fetch", START, END, len(todo))
    saved = missing = 0

    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        aliases = "\n".join(
            f"n{n}: issueOrPullRequest(number: {n}) {{ {FRAGMENT} }}" for n in chunk
        )
        query = f'{{ repository(owner: "kubernetes", name: "kubernetes") {{ {aliases} }} }}'
        for attempt in range(5):
            r = s.post(GRAPHQL, json={"query": query}, timeout=60)
            if r.status_code in (403, 429, 502, 503):
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"retries exhausted at batch offset {i}")

        payload = r.json()
        repo_data = (payload.get("data") or {}).get("repository") or {}
        for n in chunk:
            node = repo_data.get(f"n{n}")
            if not node:
                missing += 1  # deleted/transferred number; GraphQL partial errors tolerated
                continue
            (RAW_DIR / f"{n}.json").write_text(
                json.dumps(to_rest_shape(node), ensure_ascii=False), encoding="utf-8"
            )
            saved += 1
        if (i // BATCH) % 20 == 0:
            log.info(
                "batch %d/%d: saved=%d missing=%d",
                i // BATCH + 1,
                (len(todo) + BATCH - 1) // BATCH,
                saved,
                missing,
            )
        time.sleep(1.0)

    log.info("DONE. saved=%d missing=%d", saved, missing)


if __name__ == "__main__":
    main()
