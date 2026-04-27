import re


def _parse_epsg_literal(s):
    s = str(s).strip()
    m = re.search(r"(?i)\bepsg\s*:\s*(\d{4,6})\b", s) or re.match(r"^\s*(\d{4,6})\s*$", s)
    return int(m.group(1)) if m else None


def _parse_utm_any(s):
    txt = str(s).strip()
    patterns = [
        r"(?i)\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?",
        r"(?i)\bzone\s*(\d{1,2})\s*([A-Za-z])?",
        r"\b(\d{1,2})\s*([C-HJ-NP-Xc-hj-np-x])\b",
        r"\b(\d{1,2})\s*([NnSs])\b",
    ]
    m = None
    for pattern in patterns:
        m = re.search(pattern, txt)
        if m:
            break
    if not m:
        return None
    zone = int(m.group(1))
    band = (m.group(2) or "").upper()
    if band in ("N", "S"):
        hemi = band
    elif band:
        hemi = "N" if band >= "N" else "S"
    else:
        hemi = "N"
    return (32600 if hemi == "N" else 32700) + zone


def parse_crs_input(user_text: str) -> int:
    code = _parse_epsg_literal(user_text) or _parse_utm_any(user_text)
    if not code:
        raise ValueError("Invalid CRS. Try 'EPSG:32610', '32610', 'UTM 10T', or 'zone 10N'.")
    return code

