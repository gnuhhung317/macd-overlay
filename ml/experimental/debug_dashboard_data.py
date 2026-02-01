
import sys
from pathlib import Path
# Fix python path to include project root
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from ml.signal_dashboard import get_top_symbols

def debug_symbols():
    try:
        symbols_all = get_top_symbols(limit=None)
        print(f"Total symbols found (limit=None): {len(symbols_all)}")
        
        symbols_100 = get_top_symbols(limit=100)
        print(f"Total symbols found (limit=100): {len(symbols_100)}")
        
        # Check specific new coins to ensure list is fresh
        print(f"Contains 'PNUTUSDT': {'PNUTUSDT' in symbols_all}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_symbols()
