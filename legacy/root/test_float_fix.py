import sys
from typing import Any

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and invalid strings"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Test cases
test_cases = [
    (None, 0.0, 0.0),
    ("1.23", 0.0, 1.23),
    (150.5, 0.0, 150.5),
    ("invalid", 10.0, 10.0),
    ("", 0.0, 0.0),
    ({}, 5.0, 5.0),
    (None, 100.0, 100.0)
]

print("Running _safe_float verification tests...")
all_passed = True
for val, default, expected in test_cases:
    result = _safe_float(val, default)
    if result == expected:
        print(f"✅ Pass: val={val}, default={default} -> {result}")
    else:
        print(f"❌ Fail: val={val}, default={default} -> {result} (Expected {expected})")
        all_passed = False

if all_passed:
    print("\n✨ All verification tests passed successfully!")
else:
    print("\n⚠️ Some tests failed.")
    sys.exit(1)
