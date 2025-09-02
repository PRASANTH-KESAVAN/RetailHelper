"""Import Test Script"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.resolve()
paths = [
    str(project_root),
    str(project_root / "src"),
    str(project_root / "streamlit_app"),
]

for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ["PYTHONPATH"] = os.pathsep.join(paths)

print("Testing imports...")

# Test imports
tests = []

try:
    from src.utils.common import load_sample_data
    print("✅ src.utils.common imported")
    tests.append(("src.utils.common", True))
except Exception as e:
    print(f"❌ src.utils.common failed: {e}")
    tests.append(("src.utils.common", False))

try:
    from streamlit_app.components.metrics_cards import create_financial_metrics_card
    print("✅ streamlit_app.components.metrics_cards imported")
    tests.append(("metrics_cards", True))
except Exception as e:
    print(f"❌ metrics_cards failed: {e}")
    tests.append(("metrics_cards", False))

passed = sum(1 for _, success in tests if success)
total = len(tests)

print(f"\nResults: {passed}/{total} tests passed")

if passed == total:
    print("SUCCESS! You can run: streamlit run streamlit_app/dashboard_working.py")
else:
    print("Some tests failed. Check the errors above.")
