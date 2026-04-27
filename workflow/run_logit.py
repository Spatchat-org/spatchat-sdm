"""Convenience runner for logistic regression SDM."""

import os
import runpy


if __name__ == "__main__":
    method_script = os.path.join(os.path.dirname(__file__), "..", "methods", "logistic_regression.py")
    runpy.run_path(os.path.abspath(method_script), run_name="__main__")
