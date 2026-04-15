"""Batch-extract Telecom Italia snapshots for the statistical experiment.

Extracts 48 balanced snapshots from the Telecom Italia Trentino dataset:
  6 weeks  x  2 day types (weekday / weekend)  x  4 hours (08, 12, 16, 20)

Hours are UTC (Trentino is UTC+1 in November/December):
  08 UTC = 09:00 local (morning)
  12 UTC = 13:00 local (midday)
  16 UTC = 17:00 local (late afternoon)
  20 UTC = 21:00 local (evening)

Each snapshot samples 800 users from the hottest 10x10 cell patch,
normalised to [0, 2000] m.  The experiment script subsamples from these
to produce 200/400/800 user densities per seed.

Usage
-----
python scripts/extract_telecom_snapshots.py                  # extract all 48
python scripts/extract_telecom_snapshots.py --list           # print config only
python scripts/extract_telecom_snapshots.py --out-dir results/telecom_v2/snapshots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow importing from scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from extract_telecom_users import load_activity, find_hottest_crop, sample_users

# ── Balanced factorial snapshot design ────────────────────────────────────
# 4 weeks × (weekday + weekend) × 3 hours = 24 snapshots
#
# Day-of-week verification (Nov 1, 2013 = Friday):
#   Nov  3 = Sun,  Nov  4 = Mon
#   Nov  9 = Sat,  Nov 14 = Thu
#   Nov 20 = Wed,  Nov 24 = Sun
#   Dec  5 = Thu,  Dec  8 = Sun

SNAPSHOT_DESIGN = [
    # (date,          hour, week, day_type)
    # Hours are UTC: 08=morning, 12=midday, 16=late-afternoon, 20=evening
    #
    # Day-of-week verification (Nov 1, 2013 = Friday):
    #   Nov  3 = Sun,  Nov  4 = Mon
    #   Nov  9 = Sat,  Nov 14 = Thu
    #   Nov 20 = Wed,  Nov 24 = Sun
    #   Nov 25 = Mon,  Nov 30 = Sat
    #   Dec  2 = Mon,  Dec  5 = Thu,  Dec  7 = Sat,  Dec  8 = Sun
    #
    # Week 1 — early November
    ("2013-11-04",     8,   1,    "weekday"),
    ("2013-11-04",    12,   1,    "weekday"),
    ("2013-11-04",    16,   1,    "weekday"),
    ("2013-11-04",    20,   1,    "weekday"),
    ("2013-11-03",     8,   1,    "weekend"),
    ("2013-11-03",    12,   1,    "weekend"),
    ("2013-11-03",    16,   1,    "weekend"),
    ("2013-11-03",    20,   1,    "weekend"),
    # Week 2 — mid November
    ("2013-11-14",     8,   2,    "weekday"),
    ("2013-11-14",    12,   2,    "weekday"),
    ("2013-11-14",    16,   2,    "weekday"),
    ("2013-11-14",    20,   2,    "weekday"),
    ("2013-11-09",     8,   2,    "weekend"),
    ("2013-11-09",    12,   2,    "weekend"),
    ("2013-11-09",    16,   2,    "weekend"),
    ("2013-11-09",    20,   2,    "weekend"),
    # Week 3 — late November
    ("2013-11-20",     8,   3,    "weekday"),
    ("2013-11-20",    12,   3,    "weekday"),
    ("2013-11-20",    16,   3,    "weekday"),
    ("2013-11-20",    20,   3,    "weekday"),
    ("2013-11-24",     8,   3,    "weekend"),
    ("2013-11-24",    12,   3,    "weekend"),
    ("2013-11-24",    16,   3,    "weekend"),
    ("2013-11-24",    20,   3,    "weekend"),
    # Week 4 — early December
    ("2013-12-05",     8,   4,    "weekday"),
    ("2013-12-05",    12,   4,    "weekday"),
    ("2013-12-05",    16,   4,    "weekday"),
    ("2013-12-05",    20,   4,    "weekday"),
    ("2013-12-08",     8,   4,    "weekend"),
    ("2013-12-08",    12,   4,    "weekend"),
    ("2013-12-08",    16,   4,    "weekend"),
    ("2013-12-08",    20,   4,    "weekend"),
    # Week 5 — late November (second pass)
    ("2013-11-25",     8,   5,    "weekday"),
    ("2013-11-25",    12,   5,    "weekday"),
    ("2013-11-25",    16,   5,    "weekday"),
    ("2013-11-25",    20,   5,    "weekday"),
    ("2013-11-30",     8,   5,    "weekend"),
    ("2013-11-30",    12,   5,    "weekend"),
    ("2013-11-30",    16,   5,    "weekend"),
    ("2013-11-30",    20,   5,    "weekend"),
    # Week 6 — mid December
    ("2013-12-02",     8,   6,    "weekday"),
    ("2013-12-02",    12,   6,    "weekday"),
    ("2013-12-02",    16,   6,    "weekday"),
    ("2013-12-02",    20,   6,    "weekday"),
    ("2013-12-07",     8,   6,    "weekend"),
    ("2013-12-07",    12,   6,    "weekend"),
    ("2013-12-07",    16,   6,    "weekend"),
    ("2013-12-07",    20,   6,    "weekend"),
]

# Extraction parameters (fixed across all snapshots)
N_USERS     = 800       # Max density; experiment script subsamples
CROP_CELLS  = 10        # 10x10 grid cells ~ 2.35 km
AREA_SIZE_M = 2000.0    # Output coordinate range [0, area_size_m]
SEED        = 42


def extract_all(data_dir: Path, out_dir: Path, visualize: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "n_users": N_USERS,
        "crop_cells": CROP_CELLS,
        "area_size_m": AREA_SIZE_M,
        "seed": SEED,
        "n_snapshots": len(SNAPSHOT_DESIGN),
        "snapshots": [],
    }

    for idx, (date, hour, week, day_type) in enumerate(SNAPSHOT_DESIGN):
        fname = f"sms-call-internet-tn-{date}.txt"
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        print(f"  [{idx:02d}] {date} h{hour:02d} ({day_type}, week {week})  ", end="")

        activity = load_activity(str(fpath), hour_utc=hour)
        crop_row, crop_col = find_hottest_crop(activity, CROP_CELLS)

        rng = np.random.default_rng(SEED)
        positions = sample_users(
            activity, N_USERS,
            crop_row, crop_col, CROP_CELLS,
            AREA_SIZE_M, rng,
        )

        npy_name = f"snapshot_{idx:02d}.npy"
        np.save(str(out_dir / npy_name), positions)

        snap_meta = {
            "idx": idx,
            "date": date,
            "hour": hour,
            "week": week,
            "day_type": day_type,
            "crop_row": int(crop_row),
            "crop_col": int(crop_col),
            "n_active_cells": len(activity),
            "n_users_extracted": int(positions.shape[0]),
            "file": npy_name,
        }
        metadata["snapshots"].append(snap_meta)
        print(f"crop=({crop_row},{crop_col})  cells={len(activity)}  "
              f"users={positions.shape[0]}  -> {npy_name}")

    meta_path = out_dir / "snapshot_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")
    print(f"Extracted {len(metadata['snapshots'])}/{len(SNAPSHOT_DESIGN)} snapshots")

    if visualize:
        _visualize_snapshots(out_dir, metadata)


def _visualize_snapshots(out_dir: Path, metadata: dict):
    import matplotlib.pyplot as plt

    n = len(metadata["snapshots"])
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i, snap in enumerate(metadata["snapshots"]):
        pos = np.load(str(out_dir / snap["file"]))
        ax = axes[i]
        ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.3)
        ax.set_xlim(0, metadata["area_size_m"])
        ax.set_ylim(0, metadata["area_size_m"])
        ax.set_aspect("equal")
        ax.set_title(f"{snap['date']}\nh{snap['hour']:02d} {snap['day_type']}", fontsize=7)
        ax.tick_params(labelsize=5)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Telecom Italia — {n} snapshots ({metadata['n_users']} users each)")
    plt.tight_layout()
    viz_path = out_dir / "all_snapshots.png"
    plt.savefig(str(viz_path), dpi=150, bbox_inches="tight")
    print(f"Visualisation -> {viz_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="dataverse_files",
                        help="Directory with Telecom Italia .txt files (default: %(default)s)")
    parser.add_argument("--out-dir", default="results/telecom_v2/snapshots",
                        help="Output directory for .npy files + metadata (default: %(default)s)")
    parser.add_argument("--list", action="store_true",
                        help="Print snapshot design and exit")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a grid visualisation of all snapshots")
    args = parser.parse_args()

    if args.list:
        print(f"Snapshot design: {len(SNAPSHOT_DESIGN)} snapshots")
        print(f"  Users per snapshot: {N_USERS}")
        print(f"  Crop: {CROP_CELLS}x{CROP_CELLS} cells, area: {AREA_SIZE_M} m")
        print()
        for i, (date, hour, week, day_type) in enumerate(SNAPSHOT_DESIGN):
            print(f"  {i:02d}: {date}  h{hour:02d}  week={week}  {day_type}")
        return

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    print(f"Extracting {len(SNAPSHOT_DESIGN)} Telecom Italia snapshots")
    print(f"  Source:  {data_dir}")
    print(f"  Output:  {out_dir}")
    print(f"  Users:   {N_USERS}")
    print(f"  Crop:    {CROP_CELLS}x{CROP_CELLS} cells ({CROP_CELLS * 235} m)")
    print(f"  Area:    {AREA_SIZE_M} m")
    print()

    extract_all(data_dir, out_dir, visualize=args.visualize)


if __name__ == "__main__":
    main()
