"""Môi trường xe tự lái; các thuật toán học sẽ được bổ sung sau."""

import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "results" / ".matplotlib"))
