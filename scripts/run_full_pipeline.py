import os
import subprocess

# --- Read Selections ---
with open("scripts/user_layer_selection.txt", "r") as f:
    selected_layers = [line.strip() for line in f.readlines()]

with open("scripts/user_landcover_selection.txt", "r") as f:
    selected_landcover_classes = [int(line.strip()) for line in f.readlines()]

# --- Save to environment variables for fetch script ---
os.environ['SELECTED_LAYERS'] = ",".join(selected_layers)
os.environ['SELECTED_LANDCOVER_CLASSES'] = ",".join(map(str, selected_landcover_classes))

# --- Run Fetch Predictors ---
print("📥 Fetching predictors...")
subprocess.run(["python", "scripts/fetch_predictors.py"])

# --- Run Logistic SDM ---
print("🚀 Running SDM...")
subprocess.run(["python", "scripts/run_logistic_sdm.py"])