from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from analysis_tracker import get_steps


def write_session_snapshot(cache: Dict[str, Any], job_dir: str, result: Optional[dict]) -> None:
    session = cache.get("session_info") if isinstance(cache.get("session_info"), dict) else {}
    analysis_steps = get_steps(cache.get("session_id"))

    sanitized_steps: list[dict] = []
    for step in analysis_steps:
        step_copy = {k: v for k, v in step.items() if k != "code_snippets"}
        snippets = step.get("code_snippets") or []
        step_copy["code_file"] = "analysis_code.py"
        step_copy["code_labels"] = [snip.get("label") for snip in snippets]
        sanitized_steps.append(step_copy)

    snapshot: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "analysis_steps": sanitized_steps,
    }
    if isinstance(result, dict):
        snapshot["latest_result"] = result

    metadata_path = os.path.join(job_dir, "session_metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as exc:
        print(f"Warning: failed to write session metadata: {exc}")

    code_path = os.path.join(job_dir, "analysis_code.py")
    try:
        with open(code_path, "w", encoding="utf-8") as code_file:
            code_file.write("# Auto-generated snapshot of analysis code used in this session.\n")
            code_file.write("# Each section records invoked function bodies.\n\n")
            for step in analysis_steps:
                snippets = step.get("code_snippets") or []
                for snippet in snippets:
                    code_file.write(f"# ---- {snippet.get('label', 'analysis snippet')} ----\n")
                    code_file.write(snippet.get("source", ""))
                    if not snippet.get("source", "").endswith("\n"):
                        code_file.write("\n")
                    code_file.write("\n")
    except Exception as exc:
        print(f"Warning: failed to write analysis code snapshot: {exc}")


__all__ = ["write_session_snapshot"]

