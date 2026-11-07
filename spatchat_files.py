# spatchat_files.py
"""The standard "Files" tab, identical in every SpatChat room.

It works like a small directory browser:

  * one table of everything the visitor uploaded and everything SpatChat produced, with a "Remove" checkbox
    on each row (a wrong file is easy to take out), and a button to download whatever is ticked;
  * click a row to look inside it: a CSV opens as a datasheet, a chart as an image, a raster as a summary.

Spatial data also appears on the room's map; that is the room's business. This module only lists files.

A room supplies two things:
  * which files exist  -> lists of `entry(...)` dicts, usually from `scan(...)`
  * what removing one means for the room's own state (reset a dataset, a raster, ...)

Keep the copies in every room byte-identical; when the rooms merge this file moves over unchanged.
"""
from __future__ import annotations

import html
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from typing import Callable, Iterable, Optional

UPLOADED = "uploaded"
PRODUCED = "produced"

PREVIEW_ROWS = 200

# Files a room writes for its own bookkeeping; never shown to the visitor.
_HIDDEN_NAMES = {"data_metadata.json", ".gitkeep", "_spatchat_cached_df.csv", "download_results_placeholder.txt"}
_TABLE_EXTS = {".csv", ".tsv", ".txt"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_RASTER_EXTS = {".tif", ".tiff"}

COLUMNS = ["Remove", "File", "Origin", "Details", "Size", "Modified"]


# ----------------------------------------------------------------------------------------------
# entries
# ----------------------------------------------------------------------------------------------
def entry(path: str, group: str, note: str = "", name: Optional[str] = None) -> dict:
    """One file in the list. `note` is a short description such as 'Presence points' or 'Active'."""
    return {"path": os.path.abspath(path), "group": group, "note": note, "name": name or os.path.basename(path)}


def scan(root: Optional[str], group: str, *, skip_dirs: Iterable[str] = (), skip_names: Iterable[str] = (),
         exts: Optional[Iterable[str]] = None, recursive: bool = True, note_for: Optional[Callable[[str], str]] = None,
         limit: int = 400) -> list[dict]:
    """Files under `root`, newest first. `note_for(path)` can describe each one."""
    if not root or not os.path.isdir(root):
        return []
    skip_dirs = set(skip_dirs)
    skip_names = set(skip_names) | _HIDDEN_NAMES
    wanted = {e.lower() for e in exts} if exts else None
    found: list[tuple[float, dict]] = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fn in files:
            if fn in skip_names or fn.startswith("."):
                continue
            if wanted is not None and os.path.splitext(fn)[1].lower() not in wanted:
                continue
            full = os.path.join(cur, fn)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            found.append((mtime, entry(full, group, note_for(full) if note_for else "")))
        if not recursive:
            break
    found.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in found[:limit]]


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _fmt_age(ts: float) -> str:
    secs = max(0, int(time.time() - ts))
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return f"{secs // 86400} d ago"


def _rows(entries: list[dict]) -> list[list]:
    rows = []
    for e in entries:
        try:
            st = os.stat(e["path"])
            size, age = _fmt_size(st.st_size), _fmt_age(st.st_mtime)
        except OSError:
            size, age = "missing", ""
        origin = "Uploaded" if e["group"] == UPLOADED else "Produced"
        rows.append([False, e["name"], origin, e.get("note", ""), size, age])
    return rows


# ----------------------------------------------------------------------------------------------
# safe file operations
# ----------------------------------------------------------------------------------------------
def _inside(path: str, roots: Iterable[str]) -> bool:
    p = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(r)
        if p == rr or p.startswith(rr + os.sep):
            return True
    return False


def remove_paths(paths: Iterable[str], allowed_roots: Iterable[str]) -> tuple[list[str], list[str]]:
    """Delete files, but only inside `allowed_roots`. Returns (removed, problems)."""
    roots = [r for r in allowed_roots if r]
    removed, problems = [], []
    for p in paths or []:
        if not p or not _inside(p, roots):
            problems.append(f"{os.path.basename(str(p))}: not one of your files")
            continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
            removed.append(p)
        except OSError as exc:
            problems.append(f"{os.path.basename(p)}: {exc.strerror or exc}")
    return removed, problems


def zip_paths(paths: Iterable[str], allowed_roots: Iterable[str]) -> Optional[str]:
    """Zip the chosen files (or return the file itself when only one is chosen)."""
    roots = [r for r in allowed_roots if r]
    files = [p for p in (paths or []) if p and os.path.isfile(p) and _inside(p, roots)]
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    out_dir = os.path.join(tempfile.gettempdir(), "spatchat_downloads")
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"spatchat_files_{uuid.uuid4().hex[:8]}.zip")
    used: set[str] = set()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name = os.path.basename(p)
            stem, ext = os.path.splitext(name)
            i = 2
            while name in used:
                name = f"{stem}_{i}{ext}"
                i += 1
            used.add(name)
            z.write(p, arcname=name)
    return archive


# ----------------------------------------------------------------------------------------------
# previews
# ----------------------------------------------------------------------------------------------
def _read_table(path: str):
    import pandas as pd

    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, nrows=PREVIEW_ROWS * 5)
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # type: ignore[misc]


def _safe_table(df):
    import numpy as np
    import pandas as pd

    out = df.head(PREVIEW_ROWS).copy()
    out.columns = [str(c) for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        s = out[col]
        if s.dtype == object or (not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
            out[col] = s.astype(str).where(s.notna(), "")
    return out


def _raster_preview(path: str):
    """(info html, thumbnail png path or None) for a GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / 500)
        shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        data = src.read(1, out_shape=shape, masked=True).astype("float64")
        b = src.bounds
        info = (
            f"{src.width} × {src.height} cells, {src.count} band(s), CRS {src.crs}<br>"
            f"Cell size {abs(src.res[0]):.5g} × {abs(src.res[1]):.5g}, extent "
            f"{b.left:.5g}, {b.bottom:.5g} to {b.right:.5g}, {b.top:.5g}<br>NoData: {src.nodata}"
        )
    thumb = None
    if data.count():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(data, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.axis("off")
        fig.tight_layout()
        out_dir = os.path.join(tempfile.gettempdir(), "spatchat_previews")
        os.makedirs(out_dir, exist_ok=True)
        thumb = os.path.join(out_dir, f"{uuid.uuid4().hex[:8]}.png")
        fig.savefig(thumb, dpi=100)
        plt.close(fig)
    return info, thumb


def _head(name: str, kind: str, body: str = "") -> str:
    return (f'<div class="spatchat-files-preview-head"><b>{name}</b> <span>{kind}</span></div>'
            + (f'<div class="spatchat-files-preview-body">{body}</div>' if body else ""))


def _empty_table():
    """A table with one named column and no rows (a bare empty Dataframe shows stray '1 2 3' headers)."""
    import pandas as pd

    return pd.DataFrame({"(no file open)": []})


def _hint(text: str) -> str:
    return f'<div class="spatchat-files-empty">{text}</div>'


def preview(path: Optional[str]):
    """What to show for a file: (heading html, table dataframe or None, image path or None)."""
    if not path or not os.path.isfile(path):
        return _hint("Click a file above to look inside it."), None, None
    name = html.escape(os.path.basename(path))
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _TABLE_EXTS:
            df = _read_table(path)
            more = f" (first {PREVIEW_ROWS} rows shown)" if len(df) > PREVIEW_ROWS else ""
            plus = "+" if len(df) >= PREVIEW_ROWS * 5 else ""
            return _head(name, f"{len(df):,}{plus} rows × {df.shape[1]} columns{more}"), _safe_table(df), None
        if ext in _IMAGE_EXTS:
            return _head(name, "Image"), None, path
        if ext in _RASTER_EXTS:
            info, thumb = _raster_preview(path)
            return _head(name, "Raster", info), None, thumb
        if ext == ".zip":
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
            body = "<br>".join(html.escape(n) for n in names[:30]) + ("<br>…" if len(names) > 30 else "")
            return _head(name, f"Archive with {len(names)} file(s)", body), None, None
        if ext in (".json", ".geojson", ".html"):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read(1500)
            return _head(name, "Text", f"<pre>{html.escape(text)}</pre>"), None, None
    except Exception as exc:  # unreadable file: say so, do not break the tab
        return _head(name, "Could not open this file", html.escape(str(exc))[:300]), None, None
    return _head(name, f"{ext or 'File'} — no preview available"), None, None


# ----------------------------------------------------------------------------------------------
# the panel
# ----------------------------------------------------------------------------------------------
CSS = """
<style>
/* Only while the tab is the visible one: Gradio hides the others with an inline display:none. */
#spatchat-tabs #spatchat-files-pane:not([style*="display: none"]),
#spatchat-tabs #spatchat-files-pane:not([style*="display: none"]) > .column {
    display: block !important; height: auto !important; overflow-y: auto !important; overflow-x: hidden !important; }
#spatchat-tabs #spatchat-files-pane { padding: 8px 14px 14px !important; max-height: 100%; }
#spatchat-tabs #spatchat-files-pane[style*="display: none"] { display: none !important; }
#spatchat-files-pane .block { border: none !important; box-shadow: none !important; background: transparent !important;
    min-height: 0 !important; padding: 0 !important; }
#spatchat-files-pane .spatchat-files-title { font-size: 13px; font-weight: 600; color: var(--spatchat-text, #1c2b22); margin: 8px 0 4px; }
#spatchat-files-pane .spatchat-files-empty { font-size: 13px; color: var(--spatchat-text-secondary, #5c6b63);
    padding: 10px 12px; border: 1px dashed var(--spatchat-border, #cfd9d2); border-radius: 10px; margin: 4px 0; }
#spatchat-files-list { margin-bottom: 4px; }
#spatchat-files-list table { font-size: 13px; }
#spatchat-files-actions { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; margin: 8px 0 2px; align-items: center; }
#spatchat-files-actions > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#spatchat-files-actions button { white-space: nowrap !important; min-width: 130px !important; padding: 6px 14px !important; }
#spatchat-files-remove { background: transparent !important; color: #b3261e !important; border: 1px solid #b3261e !important; border-radius: 10px !important; }
#spatchat-files-status { font-size: 13px; color: var(--spatchat-text-secondary, #5c6b63); min-height: 18px; }
#spatchat-files-status .spatchat-files-problem { color: #b3261e; }
.spatchat-files-preview-head { font-size: 14px; margin: 6px 0 4px; }
.spatchat-files-preview-head span { color: var(--spatchat-text-secondary, #5c6b63); font-size: 12px; margin-left: 6px; }
.spatchat-files-preview-body { font-size: 12.5px; color: var(--spatchat-text-secondary, #5c6b63); margin-bottom: 6px; overflow-wrap: anywhere; }
.spatchat-files-preview-body pre { white-space: pre-wrap; max-height: 220px; overflow: auto; }
#spatchat-files-preview-table { margin-top: 4px; }
</style>
"""


class Panel:
    """Components of the Files tab. The room wires its own refresh/remove/download events."""

    def __init__(self, table, paths, download_btn, remove_btn, refresh_btn, status, p_info, p_table, p_image):
        self.table, self.paths = table, paths
        self.download_btn, self.remove_btn, self.refresh_btn, self.status = download_btn, remove_btn, refresh_btn, status
        self.p_info, self.p_table, self.p_image = p_info, p_table, p_image

    @property
    def outputs(self) -> list:
        """Order of the tuple returned by `updates`."""
        return [self.table, self.paths, self.download_btn, self.status, self.p_info, self.p_table, self.p_image]

    @property
    def selections(self) -> list:
        """Inputs that carry the table (with its Remove ticks) and the path behind each row."""
        return [self.table, self.paths]

    @staticmethod
    def ticked(rows, paths) -> list[str]:
        """Paths of the rows whose Remove box is ticked."""
        try:
            data = rows.values.tolist() if hasattr(rows, "values") else list(rows or [])
        except Exception:
            data = []
        return [paths[i] for i, r in enumerate(data) if i < len(paths or []) and r and bool(r[0])]

    def updates(self, gr, uploaded: list[dict], produced: list[dict], message: Optional[str] = None,
                problems: Iterable[str] = (), preview_path: Optional[str] = None, reset_preview: bool = False):
        """Fresh contents for `outputs`.

        message=None      leave the message line as it was (a plain refresh must not erase "Removed ...")
        preview_path      open that file in the preview area
        reset_preview     clear the preview (its file is gone); otherwise the preview is left alone
        """
        entries = list(uploaded) + list(produced)
        note = html.escape(message) if message else ""
        for prob in problems:
            note += f'<div class="spatchat-files-problem">{html.escape(prob)}</div>'
        status = gr.skip() if message is None else gr.update(value=note)
        if preview_path or reset_preview:
            head, table, image = preview(preview_path if preview_path else None)
            p_info, p_table, p_image = (gr.update(value=head), gr.update(value=_empty_table() if table is None else table),
                                        gr.update(value=image))
        else:
            p_info = p_table = p_image = gr.skip()
        return (
            gr.update(value=_rows(entries) or None),
            [e["path"] for e in entries],
            gr.update(value=None, interactive=False),
            status,
            p_info, p_table, p_image,
        )

    def download_update(self, gr, rows, paths, allowed_roots):
        """Point the Download button at the ticked files (zipped when there are several)."""
        path = zip_paths(self.ticked(rows, paths), allowed_roots)
        return gr.update(value=path, interactive=bool(path))

    def open_file(self, gr, paths, evt):
        """Row click: show that file below the table."""
        try:
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
            path = (paths or [])[idx]
        except Exception:
            return gr.skip(), gr.skip(), gr.skip()
        head, table, image = preview(path)
        return gr.update(value=head), gr.update(value=_empty_table() if table is None else table), gr.update(value=image)


def build(gr, *, label: str = "Files") -> Panel:
    """Build the Files tab. Call inside `with gr.Tabs(...)`. The layout is the same in every room."""
    with gr.Tab(label, id="files", elem_id="spatchat-files-pane"):
        gr.HTML(CSS)
        gr.HTML('<div class="spatchat-files-title">Your files &mdash; tick <i>Remove</i> to delete, click a row to open it</div>')
        kwargs = dict(value=None, headers=COLUMNS, datatype=["bool", "str", "str", "str", "str", "str"],
                      interactive=True, wrap=False, label="Files", show_label=False, elem_id="spatchat-files-list")
        import inspect

        accepted = set(inspect.signature(gr.Dataframe.__init__).parameters)
        # `col_count` was renamed `column_count` in Gradio 6.
        kwargs["column_count" if "column_count" in accepted else "col_count"] = (len(COLUMNS), "fixed")
        for opt, val in (("static_columns", [1, 2, 3, 4, 5]), ("max_height", 260)):
            if opt in accepted:
                kwargs[opt] = val
        table = gr.Dataframe(**kwargs)
        paths = gr.State([])
        with gr.Row(elem_id="spatchat-files-actions"):
            remove_btn = gr.Button("Remove ticked", elem_id="spatchat-files-remove", size="sm", scale=0)
            download_btn = gr.DownloadButton("Download ticked", value=None, interactive=False, size="sm", scale=0,
                                             elem_id="spatchat-files-download")
            refresh_btn = gr.Button("Refresh", size="sm", scale=0, elem_id="spatchat-files-refresh")
        status = gr.HTML(value="", elem_id="spatchat-files-status")
        p_info = gr.HTML(value=_hint("Nothing here yet. Click + in the chat or drag a file onto the panel, "
                                     "then click a file to look inside it."))
        p_table = gr.Dataframe(value=_empty_table(), interactive=False, wrap=False, show_label=False, label="Preview",
                               elem_id="spatchat-files-preview-table")
        p_image = gr.Image(value=None, type="filepath", show_label=False, interactive=False, height=300,
                           elem_id="spatchat-files-preview-image")
    panel = Panel(table, paths, download_btn, remove_btn, refresh_btn, status, p_info, p_table, p_image)

    def _open(row_paths, evt):
        return panel.open_file(gr, row_paths, evt)

    # Gradio decides to pass the click by looking at this annotation; with postponed annotations it must be set by hand.
    _open.__annotations__["evt"] = gr.SelectData

    table.select(_open, inputs=[paths], outputs=[p_info, p_table, p_image], show_progress="hidden")
    return panel


def removal_message(removed: list[str]) -> str:
    n = len(removed)
    if not n:
        return "Nothing was removed."
    names = ", ".join(os.path.basename(p) for p in removed[:3]) + ("…" if n > 3 else "")
    return f"Removed {n} file{'s' if n != 1 else ''}: {names}"
