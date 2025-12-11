#!/usr/bin/env python3
# src/parser.py
import re
import pandas as pd
from datetime import datetime
import sys
import traceback

# Optional better free-form parser
try:
    from dateutil import parser as dateutil_parser
    HAS_DATEUTIL = True
except Exception:
    HAS_DATEUTIL = False

# Candidate regex patterns (ordered). Each has a name (format key).
PATTERNS = [
    # journald / systemd ISO (your current format)
    ("iso8601", re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+\-]+)\s+(?P<host>\S+)\s+(?P<service>[\w\-.\/]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)$'
    )),
    # classic syslog: "Dec 12 10:05:23 host service[pid]: message"
    ("classic_syslog", re.compile(
        r'^(?P<timestamp>\w{3}\s+\d+\s[\d:]+)\s+(?P<host>\S+)\s+(?P<service>[\w\-.\/]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)$'
    )),
    # variant: host omitted sometimes; service:message with colon inside service portion
    ("service_colon", re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+\-]+)\s+(?P<host>\S+)\s+(?P<service>[\w\-.\/:]+):\s+(?P<message>.*)$'
    )),
    # kernel-like with bracketless service names
    ("kernel_like", re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+\-]+)\s+(?P<host>\S+)\s+(?P<service>[\S]+)\s+(?P<message>.*)$'
    )),
]

# Helper parsers for timestamps
def parse_iso(ts_str):
    try:
        # Python 3.11+: fromisoformat supports offsets; this works for "2025-12-07T03:07:28.990613+05:30"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None

def parse_classic(ts_str):
    # Add current year then parse: "Dec 12 03:07:28"
    try:
        year = datetime.now().year
        return datetime.strptime(f"{ts_str} {year}", "%b %d %H:%M:%S %Y")
    except Exception:
        return None

def parse_fallback(ts_str):
    if HAS_DATEUTIL:
        try:
            return dateutil_parser.parse(ts_str)
        except Exception:
            return None
    # Last resort: try some known formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str.split('.')[0], fmt)
        except Exception:
            continue
    return None

def try_parse_timestamp(ts_str):
    # Try iso -> classic -> dateutil fallback
    t = parse_iso(ts_str)
    if t:
        return t
    t = parse_classic(ts_str)
    if t:
        return t
    t = parse_fallback(ts_str)
    return t

def parse_line_dynamic(line):
    raw = line.rstrip("\n")
    # try all patterns
    for name, pat in PATTERNS:
        m = pat.match(raw)
        if m:
            gd = m.groupdict()
            ts_raw = gd.get("timestamp")
            ts = try_parse_timestamp(ts_raw) if ts_raw else None
            result = {
                "timestamp": ts,
                "host": gd.get("host") or "",
                "service": gd.get("service") or "",
                "pid": gd.get("pid") or "",
                "message": gd.get("message") or "",
                "raw_line": raw,
                "detected_format": name
            }
            return result
    # If no regex matched, attempt a fallback heuristic:
    # - find first token that looks like a timestamp by scanning tokens
    tokens = raw.split()
    ts_candidate = None
    # Try joining first 3 tokens and see if dateutil can parse
    if HAS_DATEUTIL:
        for end in range(1, min(6, len(tokens))+1):
            try_s = " ".join(tokens[:end])
            try:
                dt = dateutil_parser.parse(try_s, fuzzy=False)
                ts_candidate = try_s
                ts = dt
                break
            except Exception:
                continue
        if ts_candidate:
            # Heuristic: remaining text after timestamp contains host/service/message
            rest = raw[len(ts_candidate):].strip()
            parts = rest.split(None, 2)
            host = parts[0] if len(parts) > 0 else ""
            service = parts[1] if len(parts) > 1 else ""
            message = parts[2] if len(parts) > 2 else ""
            return {
                "timestamp": ts,
                "host": host,
                "service": service,
                "pid": "",
                "message": message,
                "raw_line": raw,
                "detected_format": "fallback_dateutil"
            }
    # ultimate fallback: place whole line as message
    return {
        "timestamp": None,
        "host": "",
        "service": "",
        "pid": "",
        "message": raw,
        "raw_line": raw,
        "detected_format": "unparsed"
    }

def parse_file_dynamic(infile, outfile="data/parsed_logs.csv", max_lines=None):
    rows = []
    format_counts = {}
    total = 0
    with open(infile, "r", errors="ignore") as f:
        for line in f:
            if max_lines and total >= max_lines:
                break
            total += 1
            try:
                parsed = parse_line_dynamic(line)
            except Exception:
                parsed = {"timestamp": None, "host": "", "service": "", "pid": "", "message": "", "raw_line": line.rstrip("\n"), "detected_format": "error"}
            rows.append(parsed)
            fmt = parsed.get("detected_format", "unknown")
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

    df = pd.DataFrame(rows)
    # keep original order unless timestamp exists
    if "timestamp" in df.columns and df['timestamp'].notnull().any():
        try:
            df = df.sort_values("timestamp")
        except Exception:
            pass

    df.to_csv(outfile, index=False)
    print(f"Saved parsed logs to {outfile} with {len(df)} rows.")
    # print a short summary of formats detected
    print("Format detection summary (top formats):")
    for k, v in sorted(format_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")
    return df

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "data/sample_syslog.log"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "data/parsed_logs.csv"
    max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else None
    parse_file_dynamic(infile, outfile, max_lines=max_lines)
