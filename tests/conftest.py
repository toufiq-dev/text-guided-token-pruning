"""Pytest fixtures + sys.path bootstrap so tests run with or without `pip install -e .`."""

import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC))
