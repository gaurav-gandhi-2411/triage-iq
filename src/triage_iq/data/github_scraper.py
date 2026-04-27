"""GitHub Issues scraper with rate-limit handling, pagination, and resumable caching."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GitHubScraper:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, cache_dir: str = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_repo(self, owner: str, repo: str, max_issues: int = 10000) -> int:
        """Scrape issues (+ comments) for owner/repo. Returns count of newly saved issues."""
        out_dir = self.cache_dir / f"{owner}_{repo}"
        out_dir.mkdir(parents=True, exist_ok=True)

        url = (
            f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
            f"?state=all&sort=created&direction=asc&per_page=100"
        )

        saved = 0
        seen = 0
        with tqdm(desc=f"{owner}/{repo}", unit="issue") as pbar:
            for issue in self._paginated_get(url):
                if seen >= max_issues:
                    break
                seen += 1
                number = issue["number"]
                dest = out_dir / f"{number}.json"
                if dest.exists():
                    pbar.update(1)
                    continue

                # Fetch comments and attach
                comments = self._fetch_comments(owner, repo, number)
                issue["comments_data"] = comments

                self._save_issue(dest, issue)
                saved += 1

                if saved % 100 == 0:
                    logger.info("Saved %d issues for %s/%s", saved, owner, repo)

                pbar.update(1)

        logger.info("Scrape complete. Newly saved: %d  Total seen: %d", saved, seen)
        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _paginated_get(self, url: str) -> Iterator[Dict]:
        next_url: Optional[str] = url
        while next_url:
            response = self._get_with_backoff(next_url)
            # Track rate limit from response headers (avoids extra API call)
            self._update_rate_limit_from_headers(response.headers)
            items = response.json()
            if not isinstance(items, list):
                logger.error("Unexpected response type: %s", type(items))
                break
            yield from items
            next_url = self._parse_next_link(response.headers.get("Link", ""))

    def _fetch_comments(self, owner: str, repo: str, number: int) -> list:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues/{number}/comments?per_page=100"
        comments: list = []
        for comment in self._paginated_get(url):
            comments.append(comment)
        return comments

    def _get_with_backoff(self, url: str, max_retries: int = 3) -> requests.Response:
        delay = 1
        for attempt in range(max_retries + 1):
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (500, 502, 503, 504):
                if attempt < max_retries:
                    logger.warning(
                        "HTTP %d on attempt %d — retrying in %ds",
                        resp.status_code,
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 32)
                    continue
            # Non-retryable or exhausted retries
            resp.raise_for_status()
        # unreachable, but satisfies type checker
        raise RuntimeError(f"Failed to GET {url} after {max_retries} retries")

    def _update_rate_limit_from_headers(self, headers: dict) -> None:
        """Read rate limit state from response headers; sleep if < 50 remaining."""
        remaining_str = headers.get("X-RateLimit-Remaining")
        reset_str = headers.get("X-RateLimit-Reset")
        if remaining_str is None:
            return
        remaining = int(remaining_str)
        reset_at = int(reset_str) if reset_str else int(time.time()) + 3600
        if remaining < 50:
            wait = max(0, reset_at - int(time.time())) + 5
            logger.warning(
                "Rate limit low (%d remaining). Sleeping %ds until reset.", remaining, wait
            )
            time.sleep(wait)

    def _check_rate_limit(self) -> None:
        """Explicit rate limit check via API (used only on startup or after errors)."""
        resp = self.session.get(f"{self.BASE_URL}/rate_limit", timeout=10)
        if resp.status_code != 200:
            return
        data = resp.json().get("resources", {}).get("core", {})
        remaining = data.get("remaining", 9999)
        reset_at = data.get("reset", 0)
        if remaining < 50:
            wait = max(0, reset_at - int(time.time())) + 5
            logger.warning(
                "Rate limit low (%d remaining). Sleeping %ds until reset.", remaining, wait
            )
            time.sleep(wait)

    @staticmethod
    def _save_issue(dest: Path, issue: Dict) -> None:
        dest.write_text(json.dumps(issue, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _parse_next_link(link_header: str) -> Optional[str]:
        """Parse RFC 5988 Link header and return the 'next' URL if present."""
        if not link_header:
            return None
        for part in link_header.split(","):
            url_part, *params = part.strip().split(";")
            url = url_part.strip().strip("<>")
            for param in params:
                if param.strip() == 'rel="next"':
                    return url
        return None
