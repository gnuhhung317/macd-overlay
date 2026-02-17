from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Optional
import os

app = FastAPI(title="Bitget Historical Data API")

DATA_ROOT = Path("bitget-data")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUNDING_DIR = DATA_ROOT / "funding"

class FetchRequest(BaseModel):
    limit: Optional[int] = None
    symbols: Optional[List[str]] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bitget Historical Data API"}

@app.post("/fetch")
def trigger_fetch(request: FetchRequest, background_tasks: BackgroundTasks):
    """
    Trigger the historical data fetching process in the background.
    """
    command = ["python", "bitget_fetcher.py"]
    if request.limit:
        command.extend(["--limit", str(request.limit)])
    
    # In a real app, you'd handle specific symbols differently 
    # but for now, we trigger the main script.
    
    def run_fetch():
        subprocess.run(command)

    background_tasks.add_task(run_fetch)
    return {"status": "Fetching started in background", "command": " ".join(command)}

@app.get("/symbols")
def list_symbols():
    """List all symbols with available data."""
    if not OHLCV_DIR.exists():
        return []
    files = list(OHLCV_DIR.glob("*.parquet"))
    symbols = [f.stem.replace("_USDT", "") for f in files]
    return sorted(list(set(symbols)))

@app.get("/data/ohlcv/{symbol}")
def get_ohlcv(symbol: str):
    """Retrieve OHLCV data for a specific symbol."""
    file_path = OHLCV_DIR / f"{symbol}_USDT.parquet"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Symbol data not found")
    
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")

@app.get("/data/funding/{symbol}")
def get_funding(symbol: str):
    """Retrieve funding rate data for a specific symbol."""
    file_path = FUNDING_DIR / f"{symbol}_USDT.parquet"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Symbol data not found")
    
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
