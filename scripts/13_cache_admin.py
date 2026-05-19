"""LLM response cache admin CLI.

Usage:
  python scripts/13_cache_admin.py stats
  python scripts/13_cache_admin.py clear
  python scripts/13_cache_admin.py clear-provider groq
  python scripts/13_cache_admin.py clear-model groq llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triage_iq.cache import LLMCache

DEFAULT_CACHE_PATH = ROOT / "data" / "llm_cache.sqlite"


def cmd_stats(cache: LLMCache) -> None:
    st = cache.stats()
    print(f"Path:          {st['path']}")
    print(f"Entries:       {st['entries']}")
    print(f"Total hits:    {st['total_hits_ever']}")
    print(f"Size:          {st['size_bytes']:,} bytes ({st['size_bytes'] // 1024} KB)")


def cmd_clear(cache: LLMCache) -> None:
    n = cache.clear()
    print(f"Cleared {n} entries.")


def cmd_clear_provider(cache: LLMCache, provider: str) -> None:
    n = cache.clear_provider(provider)
    print(f"Cleared {n} entries for provider '{provider}'.")


def cmd_clear_model(cache: LLMCache, provider: str, model: str) -> None:
    n = cache.clear_model(provider, model)
    print(f"Cleared {n} entries for {provider}/{model}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM response cache admin")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to cache DB (default: {DEFAULT_CACHE_PATH})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Show cache statistics")

    sub.add_parser("clear", help="Delete all cache entries")

    p_prov = sub.add_parser("clear-provider", help="Delete entries for a specific provider")
    p_prov.add_argument("provider", help="Provider name (e.g. groq, cohere, google)")

    p_model = sub.add_parser("clear-model", help="Delete entries for a specific provider+model pair")
    p_model.add_argument("provider", help="Provider name")
    p_model.add_argument("model", help="Model name")

    args = parser.parse_args()

    if not args.cache_path.exists() and args.command != "stats":
        print(f"Cache file not found: {args.cache_path}")
        sys.exit(1)

    cache = LLMCache(path=args.cache_path)
    try:
        if args.command == "stats":
            cmd_stats(cache)
        elif args.command == "clear":
            cmd_clear(cache)
        elif args.command == "clear-provider":
            cmd_clear_provider(cache, args.provider)
        elif args.command == "clear-model":
            cmd_clear_model(cache, args.provider, args.model)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
