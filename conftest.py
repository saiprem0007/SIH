# Root-level conftest.py — ensures the local src/ is always on sys.path
# regardless of any stale pip-installed versions of 'anc'.
import sys
from pathlib import Path

# Insert our local src/ FIRST so it shadows any installed package
_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
