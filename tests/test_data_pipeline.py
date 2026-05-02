"""Tests for GitHubScraper and preprocessing pipeline."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.data.github_scraper import GitHubScraper
from triage_iq.data.preprocess import clean_text, load_raw_issues, normalize_labels
from triage_iq.data.splits import stratified_classifier_split, time_based_split

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_TOKEN = "ghp_faketoken1234"


def make_issue(number: int, **kwargs) -> dict:
    return {
        "id": 1000 + number,
        "number": number,
        "title": f"Test issue {number}",
        "body": "Some body text",
        "labels": [],
        "state": "open",
        "created_at": "2024-01-01T00:00:00Z",
        "closed_at": None,
        "user": {"login": "testuser"},
        "assignees": [],
        "comments": 0,
        **kwargs,
    }


def make_response(data, status=200, link_header=""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = data
    resp.headers = {"Link": link_header}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# GitHubScraper tests
# ---------------------------------------------------------------------------

class TestGitHubScraper:

    def test_auth_header_set(self):
        scraper = GitHubScraper(token=FAKE_TOKEN)
        assert scraper.session.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"

    def test_pagination_follows_next_link(self, tmp_path):
        """Scraper should follow Link: next headers across pages."""
        issues_page1 = [make_issue(i) for i in range(1, 4)]
        issues_page2 = [make_issue(i) for i in range(4, 6)]
        comments_resp = make_response([])

        page1_url = "https://api.github.com/repos/o/r/issues?state=all&sort=created&direction=asc&per_page=100&page=1"
        page2_url = "https://api.github.com/repos/o/r/issues?page=2"
        rate_url = "https://api.github.com/rate_limit"

        def side_effect(url, timeout=30):
            if url == rate_url:
                return make_response({"resources": {"core": {"remaining": 4999, "reset": int(time.time()) + 3600}}})
            if url == page1_url:
                return make_response(
                    issues_page1,
                    link_header=f'<{page2_url}>; rel="next"',
                )
            if url == page2_url:
                return make_response(issues_page2)
            # comments
            return comments_resp

        scraper = GitHubScraper(token=FAKE_TOKEN, cache_dir=str(tmp_path))
        with patch.object(scraper.session, "get", side_effect=side_effect):
            count = scraper.scrape_repo("o", "r", max_issues=100)

        assert count == 5
        saved = list((tmp_path / "o_r").glob("*.json"))
        assert len(saved) == 5

    def test_caching_skips_existing_files(self, tmp_path):
        """Issues already saved should be skipped (resumable)."""
        out_dir = tmp_path / "o_r"
        out_dir.mkdir()
        issue = make_issue(1)
        (out_dir / "1.json").write_text(json.dumps(issue))

        issues_page = [issue]
        rate_resp = make_response({"resources": {"core": {"remaining": 4999, "reset": int(time.time()) + 3600}}})
        comments_resp = make_response([])

        call_count = {"n": 0}

        def side_effect(url, timeout=30):
            if "rate_limit" in url:
                return rate_resp
            if "comments" in url:
                return comments_resp
            call_count["n"] += 1
            return make_response([])  # empty page on second call

        # Pre-seed one issue then scrape — it must be skipped
        scraper = GitHubScraper(token=FAKE_TOKEN, cache_dir=str(tmp_path))

        def side_effect2(url, timeout=30):
            if "rate_limit" in url:
                return rate_resp
            if "comments" in url:
                return comments_resp
            return make_response(issues_page)  # single page

        with patch.object(scraper.session, "get", side_effect=side_effect2):
            newly_saved = scraper.scrape_repo("o", "r", max_issues=100)

        assert newly_saved == 0  # already exists, nothing saved

    def test_rate_limit_triggers_sleep(self, tmp_path):
        """If remaining < 50, scraper should sleep until reset."""
        reset_time = int(time.time()) + 2
        rate_resp = make_response({
            "resources": {"core": {"remaining": 10, "reset": reset_time}}
        })
        page_resp = make_response([])

        scraper = GitHubScraper(token=FAKE_TOKEN, cache_dir=str(tmp_path))

        with patch.object(scraper.session, "get", side_effect=[rate_resp, page_resp]), patch("time.sleep") as mock_sleep:
            scraper._check_rate_limit()
            mock_sleep.assert_called_once()
            wait_arg = mock_sleep.call_args[0][0]
            assert wait_arg >= 0

    def test_parse_next_link_returns_url(self):
        header = '<https://api.github.com/repos/o/r/issues?page=2>; rel="next", <https://api.github.com/repos/o/r/issues?page=10>; rel="last"'
        result = GitHubScraper._parse_next_link(header)
        assert result == "https://api.github.com/repos/o/r/issues?page=2"

    def test_parse_next_link_no_next(self):
        header = '<https://api.github.com/repos/o/r/issues?page=1>; rel="prev"'
        assert GitHubScraper._parse_next_link(header) is None

    def test_parse_next_link_empty(self):
        assert GitHubScraper._parse_next_link("") is None


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_removes_html_comments(self):
        body = "Hello <!-- this is a comment --> world"
        clean, _ = clean_text(body)
        assert "<!--" not in clean
        assert "Hello" in clean
        assert "world" in clean

    def test_extracts_code_blocks(self):
        body = "Before\n```python\nprint('hello')\n```\nAfter"
        clean, code = clean_text(body)
        assert "print('hello')" in code
        assert "[CODE_BLOCK]" in clean

    def test_truncates_long_body(self):
        body = "x" * 20_000
        clean, _ = clean_text(body)
        assert len(clean) <= 10_000

    def test_collapses_whitespace(self):
        body = "Hello\n\n\n\nWorld"
        clean, _ = clean_text(body)
        assert "\n\n\n" not in clean


class TestNormalizeLabels:

    def test_kubernetes_component_extraction(self):
        labels = ["area/kubelet", "kind/bug", "priority/critical-urgent"]
        result = normalize_labels("kubernetes_kubernetes", labels)
        assert result["component"] == "kubelet"
        assert result["type"] == "bug"
        assert result["priority"] == "critical-urgent"

    def test_unknown_repo_returns_none_facets(self):
        result = normalize_labels("unknown_repo", ["some-label"])
        assert result["component"] is None
        assert result["type"] is None
        assert result["priority"] is None

    def test_empty_labels(self):
        result = normalize_labels("kubernetes_kubernetes", [])
        assert result == {"component": None, "type": None, "priority": None}


class TestLoadRawIssues:

    def test_raises_on_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_issues("nonexistent_repo", cache_dir=str(tmp_path))

    def test_loads_and_computes_resolution_hours(self, tmp_path):
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        issue = make_issue(
            1,
            state="closed",
            created_at="2024-01-01T00:00:00Z",
            closed_at="2024-01-02T12:00:00Z",
        )
        (repo_dir / "1.json").write_text(json.dumps(issue))

        df = load_raw_issues("test_repo", cache_dir=str(tmp_path))
        assert len(df) == 1
        assert abs(df.iloc[0]["resolution_hours"] - 36.0) < 0.01

    def test_skips_malformed_json(self, tmp_path):
        repo_dir = tmp_path / "bad_repo"
        repo_dir.mkdir()
        (repo_dir / "1.json").write_text("{bad json}")
        (repo_dir / "2.json").write_text(json.dumps(make_issue(2)))

        df = load_raw_issues("bad_repo", cache_dir=str(tmp_path))
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

def _make_split_df(n: int, labels=None):
    from datetime import timezone

    import pandas as pd

    base = pd.Timestamp("2024-01-01", tz=timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            "number": i + 1,
            "closed_at": base + pd.Timedelta(hours=i),
            "component": labels[i % len(labels)] if labels else f"comp_{i % 3}",
        })
    return pd.DataFrame(rows)


class TestTimeBasedSplit:

    def test_sizes_sum_to_total(self):
        df = _make_split_df(100)
        train, val, test = time_based_split(df, 0.8, 0.1, 0.1)
        assert len(train) + len(val) + len(test) == 100

    def test_train_before_val_before_test(self):
        df = _make_split_df(100)
        train, val, test = time_based_split(df, 0.8, 0.1, 0.1)
        assert train["closed_at"].max() <= val["closed_at"].min()
        assert val["closed_at"].max() <= test["closed_at"].min()

    def test_open_issues_excluded(self):
        import pandas as pd
        df = _make_split_df(90)
        open_row = pd.DataFrame([{"number": 999, "closed_at": None, "component": "x"}])
        df = pd.concat([df, open_row], ignore_index=True)
        train, val, test = time_based_split(df, 0.8, 0.1, 0.1)
        assert len(train) + len(val) + len(test) == 90

    def test_approximate_fractions(self):
        df = _make_split_df(1000)
        train, val, test = time_based_split(df, 0.8, 0.1, 0.1)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100


class TestStratifiedClassifierSplit:

    def test_sizes_sum_to_labeled_total(self):
        df = _make_split_df(300, labels=["A", "B", "C"])
        train, val, test = stratified_classifier_split(df, "component")
        assert len(train) + len(val) + len(test) == 300

    def test_label_distribution_preserved(self):
        df = _make_split_df(300, labels=["A", "A", "B", "B", "B", "C"])
        train, val, test = stratified_classifier_split(df, "component")
        for split in (train, val, test):
            classes = set(split["component"].unique())
            assert classes == {"A", "B", "C"}

    def test_small_classes_dropped(self):
        import pandas as pd
        labels = ["A"] * 100 + ["B"] * 100 + ["rare"] * 3
        df = _make_split_df(203, labels=labels)
        train, val, test = stratified_classifier_split(
            df, "component", min_class_samples=10
        )
        all_data = pd.concat([train, val, test])
        assert "rare" not in all_data["component"].values

    def test_null_labels_excluded(self):
        df = _make_split_df(200, labels=["A", "B"])
        df.loc[0, "component"] = None
        train, val, test = stratified_classifier_split(df, "component")
        assert len(train) + len(val) + len(test) == 199
