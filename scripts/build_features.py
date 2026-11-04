"""Build and cache the feature tables in data/processed (parquet). ~5-10 minutes, ~8 GB RAM peak."""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.data import clean_application, read
from src.features import BUILDERS, application_features

OUT = Path("data/processed")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = read("application_train"), read("application_test")
    app = clean_application(pd.concat([tr, te], ignore_index=True))
    app = pd.concat([app, application_features(app)], axis=1)
    app.to_parquet(OUT / "application.parquet")
    print("application", app.shape, flush=True)
    for name, fn in BUILDERS.items():
        t0 = time.time()
        f = fn()
        f.to_parquet(OUT / f"{name}.parquet")
        print(name, f.shape, f"{time.time() - t0:.0f}s", flush=True)
        del f
        gc.collect()
