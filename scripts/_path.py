"""Makes ``activemri`` and the sibling script modules importable when a script is run directly.

Each scripts/<topic>/ folder holds a symlink to this file, so ``import _path`` resolves from
any script regardless of cwd. It puts the repo root on sys.path (for ``activemri``) and appends
every scripts/<topic>/ folder, so scripts can import each other by bare module name
(``from plot_mean_masks import load_probability``) wherever they live.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _topic in sorted((_ROOT / 'scripts').iterdir()):
    if _topic.is_dir() and not _topic.name.startswith(('_', '.')) and str(_topic) not in sys.path:
        sys.path.append(str(_topic))
