import sys
from pathlib import Path

# examples/ is not an installed package: put the example dir on sys.path so
# tests can `from response_schema import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "openai_server"))
