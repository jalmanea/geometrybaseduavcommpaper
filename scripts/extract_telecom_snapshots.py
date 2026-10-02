"""Batch-extract Telecom Italia snapshots for the statistical experiment.

Extracts 24 balanced snapshots from a Telecom Italia dataset (Milan by default):
  4 weeks  x  2 day types (weekday / weekend)  x  3 hours (10, 15, 20 UTC)

Hours are UTC (Italy is UTC+1 in November/December):
  10 UTC = 11:00 local (morning)
  15 UTC = 16:00 local (afternoon)
  20 UTC = 21:00 local (evening)

Each snapshot samples 800 users from the hottest 10x10 cell patch,
normalised to [0, 2000] m.  The experiment script subsamples from these
to produce the 100..800 user densities per seed.

Usage
-----
python scripts/extract_telecom_snapshots.py                  # Milan, all 24
python scripts/extract_telecom_snapshots.py --list           # print config only

The raw activity files are not redistributed here; see the README for the
download (Harvard Dataverse doi:10.7910/DVN/EGZHFV) and their md5 checksums.
The ``trentino_legacy`` region reproduces a superseded Trentino run that used
a wrong grid mapping; its results are not part of this repository.
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
from extract_telecom_users import REGIONS, load_activity, find_hottest_crop, sample_users

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
    # Week 1 — early November
    ("2013-11-04",    10,   1,    "weekday"),
    ("2013-11-04",    15,   1,    "weekday"),
    ("2013-11-04",    20,   1,    "weekday"),
    ("2013-11-03",    10,   1,    "weekend"),
    ("2013-11-03",    15,   1,    "weekend"),
    ("2013-11-03",    20,   1,    "weekend"),
    # Week 2 — mid November
    ("2013-11-14",    10,   2,    "weekday"),
    ("2013-11-14",    15,   2,    "weekday"),
    ("2013-11-14",    20,   2,    "weekday"),
    ("2013-11-09",    10,   2,    "weekend"),
    ("2013-11-09",    15,   2,    "weekend"),
    ("2013-11-09",    20,   2,    "weekend"),
    # Week 3 — late November
    ("2013-11-20",    10,   3,    "weekday"),
    ("2013-11-20",    15,   3,    "weekday"),
    ("2013-11-20",    20,   3,    "weekday"),
    ("2013-11-24",    10,   3,    "weekend"),
    ("2013-11-24",    15,   3,    "weekend"),
    ("2013-11-24",    20,   3,    "weekend"),
    # Week 4 — mid December
    ("2013-12-05",    10,   4,    "weekday"),
    ("2013-12-05",    15,   4,    "weekday"),
    ("2013-12-05",    20,   4,    "weekday"),
    ("2013-12-08",    10,   4,    "weekend"),
    ("2013-12-08",    15,   4,    "weekend"),
    ("2013-12-08",    20,   4,    "weekend"),
]

# Extraction parameters (fixed across all snapshots)
N_USERS     = 800       # Max density; experiment script subsamples
CROP_CELLS  = 10        # 10x10 grid cells ~ 2.35 km
AREA_SIZE_M = 2000.0    # Output coordinate range [0, area_size_m]
SEED        = 42


def extract_all(data_dir: Path, out_dir: Path, visualize: bool = False,
                region_name: str = "milan"):
    region = REGIONS[region_name]
    grid_cols, cell_size_m = region["grid_cols"], region["cell_size_m"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "n_users": N_USERS,
        "crop_cells": CROP_CELLS,
        "area_size_m": AREA_SIZE_M,
        "seed": SEED,
        "region": region_name,
        "grid_cols": grid_cols,
        "cell_size_m": cell_size_m,
        "n_snapshots": len(SNAPSHOT_DESIGN),
        "snapshots": [],
    }

    for idx, (date, hour, week, day_type) in enumerate(SNAPSHOT_DESIGN):
        fname = f"sms-call-internet-{region['file_prefix']}-{date}.txt"
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        print(f"  [{idx:02d}] {date} h{hour:02d} ({day_type}, week {week})  ", end="")

        activity = load_activity(str(fpath), hour_utc=hour)
        crop_row, crop_col = find_hottest_crop(activity, CROP_CELLS, grid_cols)

        rng = np.random.default_rng(SEED)
        positions = sample_users(
            activity, N_USERS,
            crop_row, crop_col, CROP_CELLS,
            AREA_SIZE_M, rng, grid_cols, cell_size_m,
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
    parser.add_argument("--region", choices=sorted(REGIONS), default="milan",
                        help="Grid geometry and file prefix (default: %(default)s)")
    parser.add_argument("--data-dir", default="dataverse_files_milan",
                        help="Directory with Telecom Italia .txt files (default: %(default)s)")
    parser.add_argument("--out-dir", default="results/telecom_milan/snapshots",
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
    cell = REGIONS[args.region]["cell_size_m"]
    print(f"  Region:  {args.region}")
    print(f"  Crop:    {CROP_CELLS}x{CROP_CELLS} cells ({CROP_CELLS * cell:.0f} m)")
    print(f"  Area:    {AREA_SIZE_M} m")
    print()

    extract_all(data_dir, out_dir, visualize=args.visualize, region_name=args.region)


if __name__ == "__main__":
    main()
