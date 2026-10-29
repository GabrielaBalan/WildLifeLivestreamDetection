import os
import json
import argparse

import pandas as pd

from config import OUTPUT_DIR
from dashboard_data import build_tables

TEMPLATE_PATH = "dashboard_template.html"
DATA_PLACEHOLDER = "__DASHBOARD_DATA__"


def to_records(df):
    """DataFrame -> list of dicts with timestamps as 'YYYY-MM-DDTHH:MM:SS' strings."""
    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return json.loads(df.to_json(orient="records", double_precision=3))


def main():
    parser = argparse.ArgumentParser(description="Write a self-contained HTML dashboard from the CSV logs.")
    parser.add_argument("--data-dir", default=OUTPUT_DIR, help=f"folder containing the CSV logs (default: {OUTPUT_DIR})")
    parser.add_argument("--out", help="output HTML path (default: <data-dir>/dashboard.html)")
    args = parser.parse_args()

    tables = build_tables(args.data_dir)
    data = {name: to_records(df) for name, df in tables.items()}
    data["source"] = os.path.normpath(args.data_dir)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    # "</" inside the JSON could close the <script> tag early
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out_path = args.out or os.path.join(args.data_dir, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template.replace(DATA_PLACEHOLDER, payload))

    print(f"Saved {out_path} ({os.path.getsize(out_path) // 1024} KB)")


if __name__ == "__main__":
    main()
