"""Run the packaged test suite from the repository root it expects."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]/'code'
raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q','tests','-p','no:cacheprovider'],
                                cwd=root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}))
