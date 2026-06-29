# Makes the project root importable from tests.
#
# pytest discovers this file (it sits at the repo root) and imports it before
# collecting tests, which puts the repo root on sys.path. That's what lets the
# test modules do `from pipeline import ...` and `from metrics import ...` even
# though those files live in the root, not under tests/. The explicit insert
# below makes that guarantee independent of pytest's import-mode settings.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))