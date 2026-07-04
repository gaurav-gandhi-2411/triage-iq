from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EVAL = Path(__file__).parent

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_EVAL))
sys.path.insert(0, str(_ROOT / "scripts"))
