import os
import pandas as pd
import requests
import osmnx as ox
import pickle
import random
from PIL import Image
from io import BytesIO
from pathlib import Path

# --- CONFIGURATION ---
MAPBOX_TOKEN = <>
CITY_NAME = "New York City, New York, USA"

RAW_DIR = Path("D:/Abhineet/nyc_raw_images")
CHECKPOINT_FILE = "D:/Abhineet/nyc_download_progress.pkl"

BATCH_SIZE = 1000
MAX_IMAGES =  49880#

RESET = False

RAW_DIR.mkdir(exist_ok=True)

# --- GET INTERSECTIONS ---
def get_intersections():
    
    # DROP DUPLICATES BUT KEEP FIRST OCCURRENCE (preserves order)
    df_unique = df_final.drop_duplicates(subset=["intersection_lat", "intersection_lon"], keep="first")

    coords = list(zip(df_unique["intersection_lat"], df_unique["intersection_lon"], df_unique.index))

    # LIMIT WITHOUT RANDOM SAMPLING (preserve order)
    if len(coords) > MAX_IMAGES:
        print(f"Limiting to first {MAX_IMAGES} intersections...")
        coords = coords[:MAX_IMAGES]

    return coords

# --- PIPELINE ---
def run_batch_pipeline(coords_list, token):
    results = []

    # --- RESET / RESUME LOGIC ---
    if RESET:
        print("RESET enabled: clearing checkpoint and images...")

        # Delete checkpoint
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

        # Delete images
        if RAW_DIR.exists():
            import shutil
            shutil.rmtree(RAW_DIR)
            RAW_DIR.mkdir()

        results = []

    elif os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            results = pickle.load(f)
        print(f"Resuming. {len(results)} already processed.")
    else:
        results = []

    start_idx = len(results)

    # EXTRA SAFETY LIMIT
    total_limit = min(len(coords_list), MAX_IMAGES)

    for i in range(start_idx, total_limit, BATCH_SIZE):
        batch = coords_list[i : i + BATCH_SIZE]
        print(f"Processing Batch {i // BATCH_SIZE + 1}...")

        for lat, lon, node_id in batch:
            # NAME USING LAT-LON (rounded to avoid insane filenames)
            lat_str = f"{lat:.6f}"
            lon_str = f"{lon:.6f}"
            raw_path = RAW_DIR / f"{lat_str}_{lon_str}.png"
            status = "Pending"

            if not raw_path.exists():
                url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},18,0,0/224x224?access_token={token}"
                try:
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        img = Image.open(BytesIO(res.content))
                        img.save(raw_path)
                        status = "Downloaded"
                    else:
                        status = f"HTTP {res.status_code}"
                except Exception as e:
                    status = f"Request failed: {e}"
            else:
                status = "Exists"

            results.append({
                "node_id": node_id,
                "lat": lat,
                "lon": lon,
                "raw_img": str(raw_path),
                "status": status
            })

        # --- SAVE CHECKPOINT ---
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(results, f)

    return pd.DataFrame(results)

# --- EXECUTE ---
coords = get_intersections()
df_result = run_batch_pipeline(coords, MAPBOX_TOKEN)
