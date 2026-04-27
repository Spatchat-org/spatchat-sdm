"""Backward-compatible alias for logistic full pipeline run."""

import os
import runpy
import sys


if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "run_pipeline.py")
    argv0 = sys.argv[0]
    sys.argv = [argv0, "--method", "logistic_regression", "--fetch-predictors"]
    runpy.run_path(os.path.abspath(script), run_name="__main__")
