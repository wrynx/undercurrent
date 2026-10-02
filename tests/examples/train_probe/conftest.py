import sys
from pathlib import Path

# examples/ is not an installed package: put the example dir on sys.path so
# tests can `import train_probe, linear_probe, ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "train_probe"))
