# =====================
# run_full_pipeline.py
# =====================

import sys
from scripts.predictor_fetcher import fetch_predictors
from scripts.sdm_runner import run_logistic_sdm

if len(sys.argv) < 2:
    print("Usage: python run_full_pipeline.py <layer1> <layer2> ... [--lc <landcover_class1> <landcover_class2> ...]")
    sys.exit(1)

if "--lc" in sys.argv:
    lc_index = sys.argv.index("--lc")
    selected_layers = sys.argv[1:lc_index]
    landcover_classes = list(map(int, sys.argv[lc_index + 1:]))
else:
    selected_layers = sys.argv[1:]
    landcover_classes = []

csv_path = "predictor_rasters/presence_points.csv"
fetch_predictors(csv_path, selected_layers, landcover_classes)
run_logistic_sdm(csv_path)
