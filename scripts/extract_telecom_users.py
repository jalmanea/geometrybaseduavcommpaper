"""Extract user positions from a Telecom Italia Trentino activity file.

The dataset (doi:10.7910/DVN/QLCABU) records SMS/call/internet activity in
10-minute bins for each 235×235 m grid cell in the province of Trentino.

This script:
  1. Loads one day file and sums activity over a chosen hour window.
  2. Maps square IDs to (x, y) metric coordinates using the 116-column grid.
  3. Crops the hottest N_CROP × N_CROP cells (or a fixed crop by row/col).
  4. Samples N user points inside each cell, weighted by total activity.
  5. Normalises positions to [0, area_size_m] and writes them as a .npy file.

The .npy output is shape (N, 3) with z = 0, ready for the dronecomm simulation.

Usage
-----
python scripts/extract_telecom_users.py                           # defaults
python scripts/extract_telecom_users.py --date 2013-11-07         # weekday
python scripts/extract_telecom_users.py --hour 12                 # noon UTC
python scripts/extract_telecom_users.py --n-users 400 --crop 12   # 400 pts in 12×12 patch
python scripts/extract_telecom_users.py --visualize               # show activity map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Grid constants (Trentino, 235 m cells) ─────────────────────────────────
GRID_COLS = 116          # bounding-box width in cells
CELL_SIZE_M = 235.0      # metres per cell side

# Column indices (0-based, tab-separated, some cells may be empty)
COL_SQUARE   = 0
COL_TIME_MS  = 1
# COL_COUNTRY  = 2   (not needed)
COL_SMS_IN   = 3
COL_SMS_OUT  = 4
COL_CALL_IN  = 5
COL_CALL_OUT = 6
COL_INTERNET = 7


def _safe_float(s: str) -> float:
    return float(s) if s.strip() else 0.0


def load_activity(
    path: str | Path,
    hour_utc: int | None = None,
) -> dict[int, float]:
    """Return {square_id: total_activity} for the given UTC hour (or whole day)."""
    activity: dict[int, float] = {}
    hour_ms_start = hour_utc * 3_600_000 if hour_utc is not None else None
    hour_ms_end   = hour_ms_start + 3_600_000 if hour_ms_start is not None else None

    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sq_id = int(parts[COL_SQUARE])
            t_ms  = int(parts[COL_TIME_MS])

            # Filter by hour if requested
            if hour_ms_start is not None:
                # Timestamps are epoch-ms; pick the hours-of-day component
                t_in_day = t_ms % 86_400_000
                if not (hour_ms_start <= t_in_day < hour_ms_end):
                    continue

            total = sum(
                _safe_float(parts[c]) for c in (COL_SMS_IN, COL_SMS_OUT,
                                                  COL_CALL_IN, COL_CALL_OUT,
                                                  COL_INTERNET)
                if c < len(parts)
            )
            activity[sq_id] = activity.get(sq_id, 0.0) + total

    return activity


def sq_id_to_xy(sq_id: int) -> tuple[int, int]:
    """Return (row, col) for a square ID using the 116-column Trentino grid."""
    idx = sq_id - 1          # 0-based
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    return row, col


def find_hottest_crop(
    activity: dict[int, float],
    crop_cells: int,
) -> tuple[int, int]:
    """Return (row_start, col_start) of the crop_cells×crop_cells patch with
    the highest total activity, using a sliding-window approach."""
    # Build sparse grid
    rows, cols = zip(*(sq_id_to_xy(sq) for sq in activity))
    max_row = max(rows)
    max_col = max(cols)

    grid = np.zeros((max_row + 1, max_col + 1), dtype=np.float64)
    for sq_id, val in activity.items():
        r, c = sq_id_to_xy(sq_id)
        grid[r, c] = val

    # Sliding-window sum via cumsum
    H, W = grid.shape
    if H < crop_cells or W < crop_cells:
        return 0, 0

    cum = np.cumsum(np.cumsum(grid, axis=0), axis=1)

    def box_sum(r1, c1, r2, c2):
        s = cum[r2, c2]
        if r1 > 0: s -= cum[r1 - 1, c2]
        if c1 > 0: s -= cum[r2, c1 - 1]
        if r1 > 0 and c1 > 0: s += cum[r1 - 1, c1 - 1]
        return s

    best_val = -1.0
    best_r, best_c = 0, 0
    for r in range(H - crop_cells + 1):
        for c in range(W - crop_cells + 1):
            v = box_sum(r, c, r + crop_cells - 1, c + crop_cells - 1)
            if v > best_val:
                best_val = v
                best_r, best_c = r, c

    return best_r, best_c


def sample_users(
    activity: dict[int, float],
    n_users: int,
    crop_row: int,
    crop_col: int,
    crop_cells: int,
    area_size_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_users positions within the cropped region, weighted by activity.

    Returns shape (n_users, 3) with z=0, positions normalised to [0, area_size_m].
    """
    crop_end_row = crop_row + crop_cells
    crop_end_col = crop_col + crop_cells

    # Filter to crop region
    cells_in_crop: list[tuple[int, int, float]] = []  # (row, col, weight)
    for sq_id, val in activity.items():
        if val <= 0:
            continue
        r, c = sq_id_to_xy(sq_id)
        if crop_row <= r < crop_end_row and crop_col <= c < crop_end_col:
            cells_in_crop.append((r, c, val))

    if not cells_in_crop:
        raise ValueError(
            f"No active cells in crop region "
            f"rows [{crop_row}, {crop_end_row}), cols [{crop_col}, {crop_end_col}). "
            "Try a different --date, --hour, or --crop-row / --crop-col."
        )

    rows_a  = np.array([x[0] for x in cells_in_crop], dtype=np.float64)
    cols_a  = np.array([x[1] for x in cells_in_crop], dtype=np.float64)
    weights = np.array([x[2] for x in cells_in_crop], dtype=np.float64)
    weights /= weights.sum()

    # Assign users to cells proportionally, then scatter within each cell
    cell_counts = rng.multinomial(n_users, weights)

    xy_list = []
    for i, count in enumerate(cell_counts):
        if count == 0:
            continue
        # Origin of this cell in the original grid (metres)
        x0 = (cols_a[i] - crop_col) * CELL_SIZE_M
        y0 = (rows_a[i] - crop_row) * CELL_SIZE_M
        # Uniform random within the cell
        pts = rng.uniform(0, CELL_SIZE_M, size=(count, 2))
        pts[:, 0] += x0
        pts[:, 1] += y0
        xy_list.append(pts)

    xy = np.vstack(xy_list)

    # Normalise to [0, area_size_m]
    raw_extent = crop_cells * CELL_SIZE_M
    xy = xy * (area_size_m / raw_extent)
    xy = np.clip(xy, 0.0, area_size_m)

    positions = np.zeros((xy.shape[0], 3), dtype=np.float64)
    positions[:, :2] = xy
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2013-11-07",
                        help="Date string matching filename, e.g. 2013-11-07 (default: %(default)s)")
    parser.add_argument("--hour", type=int, default=11,
                        help="UTC hour to aggregate (0-23). Default: %(default)s (noon local ~UTC+1)")
    parser.add_argument("--whole-day", action="store_true",
                        help="Aggregate the entire day instead of a single hour")
    parser.add_argument("--n-users", type=int, default=200,
                        help="Number of user positions to sample (default: %(default)s)")
    parser.add_argument("--crop", type=int, default=10,
                        help="Crop size in grid cells (NxN). Default: %(default)s (~2.35 km)")
    parser.add_argument("--crop-row", type=int, default=None,
                        help="Top-left row of crop (default: auto hottest patch)")
    parser.add_argument("--crop-col", type=int, default=None,
                        help="Top-left col of crop (default: auto hottest patch)")
    parser.add_argument("--area-size", type=float, default=2000.0,
                        help="Output coordinate range [0, area_size] in metres (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="dataverse_files",
                        help="Directory containing the .txt files (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="Output .npy path (default: results/telecom_users_<date>_h<hour>.npy)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show activity heatmap and sampled user positions")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    fname = f"sms-call-internet-tn-{args.date}.txt"
    fpath = data_dir / fname
    if not fpath.exists():
        sys.exit(f"File not found: {fpath}")

    hour = None if args.whole_day else args.hour
    hour_label = "allday" if hour is None else f"h{hour:02d}"
    print(f"Loading {fname}, {'whole day' if hour is None else f'UTC hour {hour}:00'}...")

    activity = load_activity(fpath, hour_utc=hour)
    print(f"  Active cells: {len(activity)}, total activity: {sum(activity.values()):.1f}")

    # Determine crop region
    if args.crop_row is not None and args.crop_col is not None:
        crop_row, crop_col = args.crop_row, args.crop_col
        print(f"  Using fixed crop: row={crop_row}, col={crop_col}")
    else:
        crop_row, crop_col = find_hottest_crop(activity, args.crop)
        print(f"  Auto hottest {args.crop}×{args.crop} patch: row={crop_row}, col={crop_col}")

    rng = np.random.default_rng(args.seed)
    positions = sample_users(
        activity, args.n_users,
        crop_row, crop_col, args.crop,
        args.area_size, rng,
    )
    print(f"  Sampled {positions.shape[0]} user positions in [0, {args.area_size}] m")

    out_path = args.out
    if out_path is None:
        results_dir = root / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = str(results_dir / f"telecom_users_{args.date}_{hour_label}.npy")
    np.save(out_path, positions)
    print(f"  Saved → {out_path}")

    if args.visualize:
        import matplotlib.pyplot as plt

        # Build full activity grid for display
        act_items = list(activity.items())
        rs = [sq_id_to_xy(s)[0] for s, _ in act_items]
        cs = [sq_id_to_xy(s)[1] for s, _ in act_items]
        vals = [v for _, v in act_items]
        grid = np.zeros((max(rs) + 1, max(cs) + 1))
        for r, c, v in zip(rs, cs, vals):
            grid[r, c] = v

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Telecom activity — {args.date}, {hour_label}")

        ax = axes[0]
        im = ax.imshow(np.log1p(grid), origin="lower", cmap="hot", aspect="auto")
        rect = plt.Rectangle(
            (crop_col - 0.5, crop_row - 0.5), args.crop, args.crop,
            linewidth=2, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title("log(1 + activity) — cyan = crop")
        ax.set_xlabel("col"); ax.set_ylabel("row")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1]
        ax.scatter(positions[:, 0], positions[:, 1], s=3, alpha=0.5, color="C0")
        ax.set_xlim(0, args.area_size); ax.set_ylim(0, args.area_size)
        ax.set_aspect("equal")
        ax.set_title(f"Sampled users (N={positions.shape[0]})")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

        plt.tight_layout()
        viz_path = str(out_path).replace(".npy", "_viz.png")
        plt.savefig(viz_path, bbox_inches="tight")
        print(f"  Visualisation → {viz_path}")
        plt.show()


if __name__ == "__main__":
    main()
