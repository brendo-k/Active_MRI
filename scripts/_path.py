"""Makes ``activemri`` importable when a script is run directly, regardless of cwd.

Python puts a directly-run script's own directory on sys.path[0] automatically,
so ``import _path`` resolves from any scripts/*.py without needing the repo
root on PYTHONPATH or an editable install in the environment.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
