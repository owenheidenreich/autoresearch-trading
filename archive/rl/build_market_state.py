"""Build the derived v3 market-state cache."""
from __future__ import annotations

import argparse

from v3.core.market_state import DEFAULT_MARKET_STATE_PATH, build_market_state_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v3 market-state cache")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--output", default=DEFAULT_MARKET_STATE_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = build_market_state_cache(data_path=args.data, output_path=args.output, force=args.force)
    print(f"built market state -> {path}")


if __name__ == "__main__":
    main()
