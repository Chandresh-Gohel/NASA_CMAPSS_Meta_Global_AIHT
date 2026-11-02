"""Step 1 exit check: the config loads and data.window == 30. No API key needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehagent.config import load_config


def main() -> None:
    cfg = load_config()
    window = cfg.data.window
    print(f"data.window = {window}")
    assert window == 30, f"expected data.window == 30, got {window}"
    print("Step 1 OK")


if __name__ == "__main__":
    main()
