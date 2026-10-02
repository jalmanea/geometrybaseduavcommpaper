"""Print every synthetic and orientation number quoted in the ICC paper.

Covers Sec. IV-B (synthetic sweep), Sec. IV-C (UL orientation validation) and
the synthetic figures in the abstract / conclusion of paper/main.tex.
The telecom numbers (Sec. IV-D) are printed by scripts/telecom_numbers.py.

Each line shows the value computed from results/ next to the value quoted in
the paper; the script exits non-zero if any of them differ, or if a quoted
snippet is no longer present in the .tex source.

Conventions (as in the paper)
-----------------------------
- Coverage is the plain mean over all runs in the slice; every
  (distribution, N, M) cell holds exactly 20 seeds, so this equals the mean of
  per-configuration means.
- K-means / analytic per-seed values come from ``per_seed[method]`` in
  results/heuristic_50 (seed i = 1000 + i); the altitude sweep comes from
  results/sweep_addition/synthetic_sweep.csv.
- Build times: ``results[method]["build_time_s"]`` (K-means, analytic) and the
  CSV column ``build_time_s`` (sweep).
- SINR crossings use the mean of per-seed ``mean_sinr_db`` over seeds and M.
- Orientation, single drone: per task, mean over drones of
  (``per_drone[j].best_coverage_pct`` - ``baseline.coverage_pct``), then mean
  and max over the 36 tasks. Joint: ``optimized.coverage_pct`` -
  ``baseline.coverage_pct``, mean over tasks.

python scripts/paper_numbers.py
python scripts/paper_numbers.py --altitudes   # also recompute the quoted altitudes (~1 min)

The altitudes quoted in the text are not stored in the result files, so
``--altitudes`` redeploys the analytic heuristic and the sweep to recompute them.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

METHODS = ["kmeans", "analytic", "kmeans_altitude_sweep"]
NAMES = {"kmeans": "K-means", "analytic": "analytic", "kmeans_altitude_sweep": "sweep"}
DISTS = ["clustered", "hotspot", "uniform"]
GAMMA_R_DB = 3.0

# K-means and analytic orientation results come from the SLURM run, the sweep
# from the later local run that added it.
ANGLE_RUNS = {
    "kmeans": "run_46540284",
    "analytic": "run_46540284",
    "kmeans_altitude_sweep": "run_local_altsweep",
}


class Checker:
    """Print computed values next to the paper's and remember disagreements."""

    def __init__(self, tex: str | None):
        self.tex = tex
        self.n = 0
        self.bad: list[str] = []

    def __call__(self, label: str, value, paper: str, fmt: str = "{:.1f}",
                 tex: str | None = None) -> None:
        """Compare ``fmt.format(value)`` with ``paper``; ``tex`` is a snippet of
        the source that must still contain the quoted figure."""
        got = fmt.format(value)
        self.n += 1
        status = "ok"
        if got != paper:
            status = "MISMATCH"
        elif tex is not None and self.tex is not None and tex not in self.tex:
            status = "NOT IN TEX"
        if status != "ok":
            self.bad.append(f"{label}: computed {got}, paper {paper} [{status}]"
                            + (f" (snippet: {tex})" if status == "NOT IN TEX" else ""))
        print(f"  {label:<58s} {got:>8s}   paper {paper:>8s}   {status}")


# ── Loading ──────────────────────────────────────────────────────────────

def load_synthetic(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-seed rows for the three methods, and per-configuration build times."""
    rows, times = [], []
    files = sorted(glob.glob(str(results / "heuristic_50" / "task_*.json")))
    if not files or not (results / "sweep_addition" / "synthetic_sweep.csv").exists():
        sys.exit(f"no synthetic results under {results}: run the experiments first (see README)")
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        c = d["config"]
        key = dict(dist=c["distribution"], n_drones=c["n_drones"], n_users=c["target_users"])
        for m in ("kmeans", "analytic"):
            times.append(dict(**key, method=m, build_time_s=d["results"][m]["build_time_s"]))
            for i, s in enumerate(d["per_seed"][m]):
                rows.append(dict(**key, method=m, seed=1000 + i,
                                 coverage_pct=s["coverage_pct"],
                                 mean_sinr_db=s["mean_sinr_db"]))
    pub = pd.DataFrame(rows)
    sweep = pd.read_csv(results / "sweep_addition" / "synthetic_sweep.csv")
    df = pd.concat([pub, sweep[pub.columns.tolist()]], ignore_index=True)
    print(f"synthetic: {len(files)} configurations, {len(df)} runs "
          f"({len(df) // len(METHODS)} per method)")
    return df, pd.DataFrame(times), sweep


def load_orientation(results: Path) -> pd.DataFrame:
    """One row per (method, task): mean single-drone gain and joint gain."""
    rows = []
    for m, run in ANGLE_RUNS.items():
        for f in sorted(glob.glob(str(results / "angle_sweep" / run / "task_*.json"))):
            with open(f) as fh:
                d = json.load(fh)
            c, r = d["config"], d["results"][m]
            base = r["baseline"]["coverage_pct"]
            rows.append(dict(
                method=m, dist=c["distribution"], n_drones=c["n_drones"],
                n_users=c["target_users"],
                single=np.mean([p["best_coverage_pct"] - base for p in r["per_drone"]]),
                joint=r["optimized"]["coverage_pct"] - base,
            ))
    return pd.DataFrame(rows)


# ── Sections ─────────────────────────────────────────────────────────────

def synthetic(df: pd.DataFrame, times: pd.DataFrame, sweep: pd.DataFrame, chk: Checker) -> None:
    cells = df.groupby(["dist", "n_drones", "n_users", "method"]).size()
    assert (cells == 20).all(), "every (dist, N, M, method) cell must hold 20 seeds"

    overall = df.groupby("method").coverage_pct.mean()
    by_dist = df.groupby(["dist", "method"]).coverage_pct.mean().unstack()
    by_n = df.groupby(["dist", "n_drones", "method"]).coverage_pct.mean().unstack()
    km, an, sw = (overall[m] for m in METHODS)

    print("\n=== Sec. IV-B, aggregate performance (Fig. 2) ===")
    chk("mean coverage, K-means (%)", km, "50.4", tex=r"$50.4\%$")
    chk("mean coverage, analytic (%)", an, "76.7", tex=r"$76.7\%$")
    chk("mean coverage, sweep (%)", sw, "80.6", tex=r"$80.6\%$")
    chk("analytic - K-means, all configurations (pp)", an - km, "+26.3", "{:+.1f}",
        tex=r"leads by $+26.3$\,pp")
    chk("sweep - K-means, all configurations (pp)", sw - km, "+30.2", "{:+.1f}",
        tex=r"leads by $+30.2$\,pp")
    chk("sweep - analytic, all configurations (pp)", sw - an, "+3.9", "{:+.1f}",
        tex=r"by $+3.9$\,pp over the analytic")
    d = by_dist.analytic - by_dist.kmeans
    chk("analytic - K-means, clustered (pp)", d["clustered"], "+37.6", "{:+.1f}", tex=r"$+37.6$")
    chk("analytic - K-means, hotspot (pp)", d["hotspot"], "+42.5", "{:+.1f}", tex=r"$+42.5$")
    chk("uniform, K-means (%)", by_dist.kmeans["uniform"], "89.5", tex=r"$89.5\%$")
    chk("uniform, analytic (%)", by_dist.analytic["uniform"], "88.5", tex=r"$88.5\%$")
    chk("uniform, sweep (%)", by_dist.kmeans_altitude_sweep["uniform"], "92.9", tex=r"$92.9\%$")
    d = by_dist.kmeans_altitude_sweep - by_dist.analytic
    chk("sweep - analytic, clustered (pp)", d["clustered"], "+6.1", "{:+.1f}", tex=r"$+6.1$\,pp")
    chk("sweep - analytic, uniform (pp)", d["uniform"], "+4.5", "{:+.1f}", tex=r"$+4.5$\,pp")
    chk("sweep - analytic, hotspot (pp)", d["hotspot"], "+1.0", "{:+.1f}", tex=r"$+1.0$\,pp")
    print("  Fig. 2 bar labels (%):")
    print(by_dist.loc[DISTS, METHODS].round(0).astype(int).to_string().replace("\n", "\n    ").join(["    ", ""]))

    uni = by_n.loc["uniform"]
    lead = uni.analytic > uni.kmeans
    first = min(n for n in uni.index if lead.loc[n:].all())
    chk("uniform: analytic above K-means for all N >=", first, "41", "{:d}", tex=r"$N \geq 41$")
    chk("uniform: analytic - K-means at N=50 (pp)", uni.analytic[50] - uni.kmeans[50], "+3.5",
        "{:+.1f}", tex=r"$+3.5$\,pp at $N{=}50$")

    print("\n=== Sec. IV-B, build times ===")
    bt = times.groupby("method").build_time_s.mean()
    chk("mean build time, K-means (ms)", bt["kmeans"] * 1e3, "2.3", tex=r"$2.3$ and $5.0$\,ms")
    chk("mean build time, analytic (ms)", bt["analytic"] * 1e3, "5.0", tex=r"$2.3$ and $5.0$\,ms")
    chk("mean build time, sweep (ms)", sweep.build_time_s.mean() * 1e3, "135", "{:.0f}",
        tex=r"averages $135$\,ms")
    chk("max build time, sweep (s)", sweep.build_time_s.max(), "0.91", "{:.2f}", tex=r"$0.91$\,s")
    chk("sweep runs", len(sweep), "30000", "{:d}", tex=r"$30{,}000$ runs")

    print("\n=== Sec. IV-B, per-configuration breakdown (Fig. 3) ===")
    hot, clu = by_n.loc["hotspot"], by_n.loc["clustered"]
    chk("hotspot, K-means at N=5 (%)", hot.kmeans[5], "64.8", tex=r"$64.8\%$ at $N{=}5$")
    chk("hotspot, K-means at N=40 (%)", hot.kmeans[40], "0", "{:.0f}", tex=r"$0\%$ by $N{=}40$")
    chk("hotspot, analytic at N=5 (%)", hot.analytic[5], "92.0", tex=r"$92.0\%$ at $N{=}5$")
    chk("hotspot, analytic at N=50 (%)", hot.analytic[50], "22.1", tex=r"$22.1\%$ at $N{=}50$")
    chk("hotspot, sweep at N=5 (%)", hot.kmeans_altitude_sweep[5], "93.5", tex=r"$93.5\%$ and $23.3\%$")
    chk("hotspot, sweep at N=50 (%)", hot.kmeans_altitude_sweep[50], "23.3", tex=r"$93.5\%$ and $23.3\%$")

    sinr = (df[df.dist == "hotspot"].groupby(["n_drones", "method"]).mean_sinr_db.mean().unstack())
    below = sinr < GAMMA_R_DB
    for m in METHODS:
        assert below[m].loc[below[m].idxmax():].all(), "SINR crosses the threshold more than once"
    chk("hotspot: K-means mean SINR first below 3 dB at N =", int(below.kmeans.idxmax()), "11",
        "{:d}", tex=r"at $N{=}11$")
    chk("hotspot: analytic mean SINR above 3 dB up to N =", int(below.analytic.idxmax()) - 1, "33",
        "{:d}", tex=r"$N{=}33$ (analytic)")
    chk("hotspot: sweep mean SINR above 3 dB up to N =",
        int(below.kmeans_altitude_sweep.idxmax()) - 1, "34", "{:d}", tex=r"$N{=}34$ (sweep)")

    hi = clu.loc[10:]
    chk("clustered, analytic, N >= 10: min (%)", hi.analytic.min(), "80", "{:.0f}", tex=r"$80$--$88\%$")
    chk("clustered, analytic, N >= 10: max (%)", hi.analytic.max(), "88", "{:.0f}", tex=r"$80$--$88\%$")
    chk("clustered, sweep, N >= 10: min (%)", hi.kmeans_altitude_sweep.min(), "86", "{:.0f}",
        tex=r"$86$--$93\%$")
    chk("clustered, sweep, N >= 10: max (%)", hi.kmeans_altitude_sweep.max(), "93", "{:.0f}",
        tex=r"$86$--$93\%$")
    chk("clustered, K-means at N=5 (%)", clu.kmeans[5], "94.2", tex=r"$94.2\%$ at $N{=}5$")
    chk("clustered, K-means at N=50 (%)", clu.kmeans[50], "12.0", tex=r"$12.0\%$ at $N{=}50$")
    chk("clustered, K-means monotone decreasing for N >= 5",
        bool((clu.kmeans.loc[5:].diff().dropna() < 0).all()), "True", "{}",
        tex="degrades monotonically")
    chk("clustered, analytic at N=50 (%)", clu.analytic[50], "80.2", tex=r"$80.2\%$ and $86.4\%$")
    chk("clustered, sweep at N=50 (%)", clu.kmeans_altitude_sweep[50], "86.4",
        tex=r"$80.2\%$ and $86.4\%$")

    print("\n=== Sec. IV-B, scaling with fleet size ===")
    margin = by_n.kmeans_altitude_sweep - by_n[["kmeans", "analytic"]].max(axis=1)
    n = margin.index.get_level_values("n_drones")
    chk("sweep above both others at every N >= 4, all distributions",
        bool((margin[n >= 4] > 0).all()), "True", "{}", tex=r"every fleet size $N \geq 4$")
    chk("largest sweep deficit for N < 4 (pp)", max(0.0, -margin[n < 4].min()), "0.1",
        tex=r"more than $0.1$\,pp")
    chk("uniform, sweep at N=5 (%)", uni.kmeans_altitude_sweep[5], "93.0", tex=r"$93.0\%$ at $N{=}5$")
    chk("uniform, sweep at N=50 (%)", uni.kmeans_altitude_sweep[50], "93.5", tex=r"$93.5\%$ at $N{=}50$")
    net = uni.loc[50] - uni.loc[5]
    chk("uniform: methods whose coverage rises from N=5 to N=50",
        ", ".join(NAMES[m] for m in METHODS if net[m] > 0), "sweep", "{}",
        tex="the only method whose coverage")
    print("    net change N=5 -> N=50 (pp): "
          + ", ".join(f"{NAMES[m]} {net[m]:+.1f}" for m in METHODS))


def orientation(o: pd.DataFrame, chk: Checker) -> None:
    print("\n=== Sec. IV-C, UL orientation validation ===")
    counts = o.groupby("method").size()
    assert (counts == 36).all(), f"expected 36 tasks per method, got {dict(counts)}"
    paper = {
        "analytic": ("0.40", "0.78", "uniform", 5, "+3.9",
                     r"$0.40$\,pp of coverage for the analytic heuristic (max $0.78$\,pp, uniform, $N{=}5$)",
                     r"$+3.9$\,pp for the analytic heuristic"),
        "kmeans_altitude_sweep": ("0.34", "0.71", "hotspot", 25, "+3.3",
                                  r"$0.34$\,pp for the sweep (max $0.71$\,pp, hotspot, $N{=}25$)",
                                  r"$+3.3$\,pp for the sweep"),
        "kmeans": ("0.77", "4.47", "hotspot", 5, "+7.7",
                   r"$0.77$\,pp for K-means (max $4.47$\,pp, hotspot, $N{=}5$)",
                   r"$+7.7$\,pp for K-means"),
    }
    for m, (mean, mx, dist, n, joint, tex_single, tex_joint) in paper.items():
        d = o[o.method == m]
        top = d.loc[d.single.idxmax()]
        chk(f"{NAMES[m]}: single-drone gain, mean (pp)", d.single.mean(), mean, "{:.2f}", tex=tex_single)
        chk(f"{NAMES[m]}: single-drone gain, max (pp)", top.single, mx, "{:.2f}", tex=tex_single)
        chk(f"{NAMES[m]}: configuration of the max", f"{top.dist}, N={top.n_drones}",
            f"{dist}, N={n}", "{}", tex=tex_single)
        chk(f"{NAMES[m]}: joint gain, mean (pp)", d.joint.mean(), joint, "{:+.1f}", tex=tex_joint)


def abstract(df: pd.DataFrame, times: pd.DataFrame, sweep: pd.DataFrame, chk: Checker) -> None:
    print("\n=== Abstract and conclusion (synthetic) ===")
    c = df.groupby("method").coverage_pct.mean()
    chk("abstract: analytic gain over K-means (pp)", c.analytic - c.kmeans, "+26.3", "{:+.1f}",
        tex=r"by $+26.3$ and $+30.2$ percentage points")
    chk("abstract: sweep gain over K-means (pp)", c.kmeans_altitude_sweep - c.kmeans, "+30.2",
        "{:+.1f}", tex=r"by $+26.3$ and $+30.2$ percentage points")
    chk("abstract: analytic mean deployment time (ms)",
        times[times.method == "analytic"].build_time_s.mean() * 1e3, "5", "{:.0f}",
        tex=r"$5$ and $135$\,ms")
    chk("abstract: sweep mean deployment time (ms)", sweep.build_time_s.mean() * 1e3, "135",
        "{:.0f}", tex=r"$5$ and $135$\,ms")
    chk("conclusion: sweep - analytic, synthetic (pp)", c.kmeans_altitude_sweep - c.analytic,
        "+3.9", "{:+.1f}", tex=r"closed-form rule by $+3.9$\,pp")
    chk("sweep never below K-means in any (dist, N) mean",
        bool((df.groupby(["dist", "n_drones", "method"]).coverage_pct.mean().unstack()
              .eval("kmeans_altitude_sweep - kmeans") > -0.1).all()), "True", "{}",
        tex="the sweep never loses to a fixed-altitude K-means baseline")


def altitudes(results: Path, sweep: pd.DataFrame, chk: Checker) -> None:
    """Altitudes the paper quotes but the result files do not store: redeploy."""
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "scripts"))
    from dronecomm.heuristic import (deploy_analytic_heuristic,
                                     deploy_kmeans_global_altitude_sweep)
    import run_heuristic_experiment as syn
    import run_telecom_statistical as tel

    print("\n=== Altitudes (recomputed deployments) ===")
    # Uniform synthetic demand: analytic flies above the 150 m baseline for N <~ 30.
    alt = {}
    for n in range(1, 51):
        zs = []
        for m_users in range(100, 1001, 100):
            cfg = syn.make_config(n, m_users, "uniform")
            for seed in range(1000, 1020):
                users = syn.generate_users_for_distribution(
                    "uniform", m_users, cfg, np.random.default_rng(seed))
                sc = deploy_analytic_heuristic(cfg, users, seed=seed)
                zs.append(sc.drone_positions[:, 2].mean())
        alt[n] = float(np.mean(zs))
    alt = pd.Series(alt)
    last = int(alt[alt > 150.0].index.max())
    # The paper only says "N <~ 30", so this one is printed, not checked.
    print(f"  uniform, analytic mean altitude above 150 m up to N = {last} "
          f"({alt[last]:.0f} m; N={last + 1}: {alt[last + 1]:.0f} m)   paper: N <~ 30")

    # Milan, M = 800: mean altitude at N = 5 and N = 30 (nearest 10 m).
    snap_dir = results / "telecom_milan" / "snapshots"
    snaps = tel.load_snapshot_metadata(snap_dir)
    quoted = {("analytic", 5): "300", ("analytic", 30): "170",
              ("sweep", 5): "250", ("sweep", 30): "110"}
    tex = {"analytic": r"from about $300$\,m at $N{=}5$ to $170$\,m at $N{=}30$",
           "sweep": r"against about $250$ to $110$\,m for the sweep"}
    deploy = {"analytic": deploy_analytic_heuristic, "sweep": deploy_kmeans_global_altitude_sweep}
    for name, fn in deploy.items():
        for n in (5, 30):
            cfg = tel.make_config(n)
            zs = []
            for s in snaps:
                full = np.load(str(snap_dir / s["file"]))
                for seed in range(tel.BASE_SEED, tel.BASE_SEED + 20):
                    users = tel.subsample_users(full, 800, seed)
                    zs.append(fn(cfg, users, seed=seed).drone_positions[:, 2].mean())
            z = float(np.mean(zs))
            print(f"    Milan, {name}, N={n}: {z:.1f} m")
            chk(f"Milan, {name} mean altitude at N={n} (m, nearest 10)", round(z, -1),
                quoted[(name, n)], "{:.0f}", tex=tex[name])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--tex", default="paper/main.tex",
                    help="paper source; quoted snippets are looked up in it if it exists")
    ap.add_argument("--altitudes", action="store_true",
                    help="also redeploy to recompute the altitudes quoted in the text")
    args = ap.parse_args()
    results = Path(args.results)
    tex_path = Path(args.tex)
    tex = None
    if tex_path.exists():
        # Drop comment lines so that numbers surviving only in commented-out
        # drafts do not count as present.
        tex = "\n".join(l for l in tex_path.read_text().splitlines()
                        if not l.lstrip().startswith("%"))
    else:
        print(f"note: {tex_path} not found, comparing against hard-coded paper values only")
    chk = Checker(tex)

    df, times, sweep = load_synthetic(results)
    synthetic(df, times, sweep, chk)
    orientation(load_orientation(results), chk)
    abstract(df, times, sweep, chk)
    if args.altitudes:
        altitudes(results, sweep, chk)

    print(f"\n{chk.n} values checked, {len(chk.bad)} mismatch(es)")
    for b in chk.bad:
        print("  " + b)
    sys.exit(1 if chk.bad else 0)


if __name__ == "__main__":
    main()
