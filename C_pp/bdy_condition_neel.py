#!/usr/bin/env python3
"""
Extract q(x,t) and J(x,t) at a fixed interface-centered coordinate x_rel directly from
covariance files produced by main_dynamics_it_cov.py.

run example:
cd /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp
source ~/venvs/qsim/bin/activate

python3 bdy_condition_neel.py \
  --outdir ./runs/neel_cov_local/GHD_BDY \
  --cov-dir ./runs/neel_cov_local/GHD_IT_COV \
  --init-state neel \
  --sizes "1000" \
  --s-offsets "1000" \
  --gammas "0.5 1.0" \
  --times "100 150 200 250 300 400" \
  --r 3 \
  --sign - \
  --x-rel 0

"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)

from profiles_from_cov_it import *


def compute_point(C_t, x_center, r, gamma, sign):
    if sign == "-":
        q_center = float(np.real(qm_left(r, x_center, C_t, "open")))
        j_r = float(np.real(jm_left(r, x_center +1, C_t, "open")))
        j_l = float(np.real(jm_left(r, x_center -1, C_t, "open")))
        Delta_j= j_r -j_l
        rhs_term= gamma*q_center
        error_bdy= np.abs(Delta_j + rhs_term)
    else:
        return
    return Delta_j, rhs_term, error_bdy


def base_name(meta0, x_rel, r, sign):
    prefix = {
        "neel": "GHD_BDY_NEEL_COV",
        "neel_even": "GHD_BDY_NEEL_even_COV",
        "vac_fill": "GHD_BDY_fill_COV"
    }[meta0["init_state"]]
    return (
        f"{prefix}_x{x_rel}_r{r}_s_{meta0['s_offset']}_sign{sign}"
        f"_gamma{meta0['gamma']:.2f}_N{meta0['N']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Extract q(x,t), J(x,t) directly from GHD_IT_COV files.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for CSV/PNG")
    ap.add_argument("--cov-dir", type=str, default=None, help="Directory with covariance .npz files")
    ap.add_argument("--init-state", type=str, required=True, choices=["neel", "neel_even", "vac_fill" ])
    ap.add_argument("--sizes", type=str, required=True, help="Comma/space-separated half-sizes L")
    ap.add_argument("--s-offsets", type=str, required=True, help="Comma/space-separated s offsets")
    ap.add_argument("--gammas", type=str, required=True, help="Comma/space-separated gamma values")
    ap.add_argument("--times", type=str, default=None, help="Optional comma/space-separated times")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sign", type=str, required=True, choices=["+", "-"])
    ap.add_argument("--x-rel", type=int, default=0, help="Interface-centered coordinate x-(center-1)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--phsymm-m", type=int, default=None)
    ap.add_argument("--phsymm-A", type=float, default=None)

    args = ap.parse_args()

    args.sizes = parse_list(args.sizes, int)
    args.s_offsets = parse_list(args.s_offsets, int)
    args.gammas = parse_list(args.gammas, float)
    args.times = parse_list(args.times, float)

    cov_dir = build_cov_search_dir(args.outdir, args.cov_dir)
    if not os.path.isdir(cov_dir):
        raise SystemExit(f"Missing covariance directory: {cov_dir}")

    entries = []
    for root, _, files in os.walk(cov_dir):
        for fname in files:
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(root, fname)
            meta = load_meta(path)
            if matches(meta, args):
                entries.append(meta)

    if not entries:
        raise SystemExit("No matching covariance files found.")

    rows = []
    for meta in entries:
        with np.load(meta["path"], allow_pickle=False) as data:
            C_t = data["C_t"]
            center = int(data["center"])
            x_center = (center - 1) + args.x_rel
            if x_center < 0 or x_center >= meta["N"]:
                raise SystemExit(f"x-rel={args.x_rel} is outside valid range for N={meta['N']}.")
            delta_j, rhs_term, error_bdy = compute_point(C_t, x_center, args.r, meta["gamma"], args.sign)
            zeta_val = float(x_center - (center - 1)) / float(meta["time"])
            rows.append((meta["time"], zeta_val, delta_j, rhs_term, error_bdy, meta))

    # Sort by time rows[0]
    rows.sort(key=lambda item: item[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    zeta_axis = np.array([r[1] for r in rows], dtype=float)
    delta_j = np.array([r[2] for r in rows], dtype=float)
    rhs_term = np.array([r[3] for r in rows], dtype=float)
    error_bdy = np.array([r[4] for r in rows], dtype=float)
    # metadata dictionary from the first row after sorting. 
    meta0 = rows[0][5]
    x_center0 = (meta0["center"]-1) + args.x_rel

    os.makedirs(args.outdir, exist_ok=True)
    base = base_name(meta0, args.x_rel, args.r, args.sign)
    out_csv = os.path.join(args.outdir, base + ".csv")
    out_png = os.path.join(args.outdir, base + ".png")

    np.savetxt(
        out_csv,
        np.column_stack([t_axis, zeta_axis, delta_j, rhs_term,  error_bdy]),
        delimiter=",",
        header="time,zeta_at_x,delta_j,rhs_term,error_bdy",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].scatter(t_axis, error_bdy, color="tab:blue", s=8.8)
    axes[1].scatter(t_axis, abs(delta_j), color="tab:green", s=8.8, marker="s", label= r"$|\Delta J|$")
    axes[1].scatter(t_axis, abs(rhs_term), color="tab:red", s=8.8, marker="^", label =r"$|\gamma q_0|$")
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$log|\Delta J - \gamma q_0|$")
    axes[0].set_yscale("log")
    plt.legend()
    fig.suptitle(rf"$x_0={x_center0},\ r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={meta0['gamma']},\ N={meta0['N']}$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {len(rows)} time slices")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
