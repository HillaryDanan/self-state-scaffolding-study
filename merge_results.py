#!/usr/bin/env python3
"""Merge separate provider result files into one combined file."""

import json
import os
import glob
from datetime import datetime

DATA_DIR = "data"

def main():
    pattern = os.path.join(DATA_DIR, "raw_results_*.json")
    files = sorted(glob.glob(pattern))
    all_pattern = os.path.join(DATA_DIR, "all_results_*.json")
    all_files = sorted(glob.glob(all_pattern))

    print("Found result files:")
    for f in files:
        print(f"  {f}")
    for f in all_files:
        print(f"  {f}")

    combined = {}
    name_map = {
        "anthropic": "Claude Sonnet 4.6",
        "openai": "GPT-5.2",
        "google": "Gemini 3 Flash",
    }

    # Load individual provider files
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace("raw_results_", "").replace(".json", "")
        provider_key = parts.split("_")[0]
        display_name = name_map.get(provider_key, provider_key)

        with open(f) as fh:
            data = json.load(fh)
        print(f"\n  {f} -> {display_name}")
        for cond, trials in data.items():
            print(f"    {cond}: {len(trials)} trials")
        combined[display_name] = data

    # Load from all_results files too
    for f in all_files:
        with open(f) as fh:
            data = json.load(fh)
        for prov, cond_data in data.items():
            if prov not in combined:
                combined[prov] = cond_data
                print(f"    Added {prov} from {f}")

    # Save combined
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"all_results_combined_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nCOMBINED: {out_path}")
    for prov, conds in combined.items():
        total = sum(len(t) for t in conds.values())
        print(f"  {prov}: {total} trials")
    print(f"\nRun:  python3 analyze_results.py --input {out_path}")

if __name__ == "__main__":
    main()