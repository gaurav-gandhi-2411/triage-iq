"""CLI runner for the GitHub Issues scraper.

Usage:
    python scripts/01_scrape_issues.py --repo microsoft/vscode --max-issues 5000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from triage_iq.data.github_scraper import GitHubScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logging.critical("GITHUB_TOKEN not set. Add it to .env and retry.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Scrape GitHub Issues for a repository.")
    parser.add_argument("--repo", required=True, help="owner/repo e.g. microsoft/vscode")
    parser.add_argument("--max-issues", type=int, default=10_000)
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--sort", default="created", choices=["created", "updated", "comments"],
                        help="Sort field (default: created)")
    parser.add_argument("--direction", default=None, choices=["asc", "desc"],
                        help="Sort direction. Defaults to asc for created, desc for updated.")
    parser.add_argument("--since", default=None,
                        help="Only include issues updated at or after this date (YYYY-MM-DD). "
                             "Useful with --sort updated.")
    args = parser.parse_args()

    if "/" not in args.repo:
        logging.critical("--repo must be in owner/repo format")
        sys.exit(1)

    # Sensible direction default: desc for updated (newest first), asc for created
    direction = args.direction or ("desc" if args.sort == "updated" else "asc")

    owner, repo = args.repo.split("/", 1)
    scraper = GitHubScraper(token=token, cache_dir=args.cache_dir)
    saved = scraper.scrape_repo(
        owner=owner,
        repo=repo,
        max_issues=args.max_issues,
        sort=args.sort,
        direction=direction,
        since=args.since,
    )
    logging.info("Done. Newly saved issues: %d", saved)


if __name__ == "__main__":
    main()
