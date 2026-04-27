import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflow"
LAYER_SELECTION_FILE = WORKFLOW_DIR / "user_layer_selection.txt"
LANDCOVER_SELECTION_FILE = WORKFLOW_DIR / "user_landcover_selection.txt"

METHOD_TO_SCRIPT = {
    "logistic_regression": "methods/logistic_regression.py",
    "issa": "methods/issa.py",
}


def _read_lines_if_exists(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            lines.append(value)
    return lines


def _collect_layers() -> list[str]:
    env_layers = [v.strip() for v in os.environ.get("SELECTED_LAYERS", "").split(",") if v.strip()]
    if env_layers:
        return env_layers
    return _read_lines_if_exists(LAYER_SELECTION_FILE)


def _collect_landcover_codes() -> list[str]:
    env_codes = [v.strip() for v in os.environ.get("SELECTED_LANDCOVER_CLASSES", "").split(",") if v.strip()]
    if env_codes:
        return env_codes
    return _read_lines_if_exists(LANDCOVER_SELECTION_FILE)


def _run_step(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(label)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spatchat SDM workflow.")
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_TO_SCRIPT.keys()),
        default="logistic_regression",
        help="Model method to run.",
    )
    parser.add_argument(
        "--fetch-predictors",
        action="store_true",
        help="Fetch predictors before running the selected method.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    if args.fetch_predictors:
        layers = _collect_layers()
        landcover_codes = _collect_landcover_codes()
        if not layers:
            raise RuntimeError(
                "No predictor layers selected. Set SELECTED_LAYERS env var or create workflow/user_layer_selection.txt."
            )
        env["SELECTED_LAYERS"] = ",".join(layers)
        env["SELECTED_LANDCOVER_CLASSES"] = ",".join(landcover_codes)
        _run_step([sys.executable, "workflow/fetch_predictors.py"], env, "Fetching predictors...")

    method_script = METHOD_TO_SCRIPT[args.method]
    _run_step([sys.executable, method_script], env, f"Running method: {args.method}...")
    print("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(2)
