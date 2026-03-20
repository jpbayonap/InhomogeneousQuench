#!/usr/bin/env python3
"""
Extract q(x,t) and J(x,t) at a fixed interface-centered coordinate x_rel directly from
covariance files produced by main_dynamics_it_cov.py.
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

from main_dynamics_it import qp_symm, qm_symm, jp_symm, jm_symm
from profiles_from_cov_it import build_cov_search_dir, load_meta, matches, parse_list


def compute_point(C_t, x_abs, r, sign):
    if sign == "-":
        q_val = float(np.real(qm_symm(r, x_abs, C_t, "open")))
        j_val = float(np.real(jm_symm(r, x_abs, C_t, "open")))
    else:
        q_val = float(np.real(qp_symm(r, x_abs, C_t, "open")))
        j_val = float(np.real(jp_symm(r, x_abs, C_t, "open")))
    return q_val, j_val


def base_name(meta0, x_rel, r, sign):
    prefix = {
        "neel": "GHD_QXT_NEEL_COV",
        "beta": "GHD_QXT_Beta_COV",
        "beta_lr": "GHD_QXT_betaLR_COV",
        "vac_fill": "GHD_QXT_VAC_FILL_COV",
        "vac_infty": "GHD_QXT_VAC_INFTY_COV",
        "mixed_neel": "GHD_QXT_MIXED_NEEL_COV",
        "phsymm": "GHD_QXT_PHSYMM_COV",
        "phsymm_odd": "GHD_QXT_PHSYMM_ODD_COV",
    }[meta0["init_state"]]
    return (
        f"{prefix}_x{x_rel}_r{r}_s_{meta0['s_offset']}_sign{sign}"
        f"_gamma{meta0['gamma']:.2f}_N{meta0['N']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Extract q(x,t), J(x,t) directly from GHD_IT_COV files.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for CSV/PNG")
    ap.add_argument("--cov-dir", type=str, default=None, help="Directory with covariance .npz files")
    ap.add_argument("--init-state", type=str, required=True, choices=["neel", "beta", "beta_lr", "vac_fill", "mixed_neel", "vac_infty", "phsymm", "phsymm_odd"])
    ap.add_argument("--sizes", type=str, required=True, help="Comma/space-separated half-sizes L")
    ap.add_argument("--s-offsets", type=str, required=True, help="Comma/space-separated s offsets")
    ap.add_argument("--gammas", type=str, required=True, help="Comma/space-separated gamma values")
    ap.add_argument("--times", type=str, default=None, help="Optional comma/space-separated times")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--phsymm-m", type=int, default=None)
    ap.add_argument("--phsymm-A", type=float, default=None)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sign", type=str, required=True, choices=["+", "-"])
    ap.add_argument("--x-rel", type=int, default=0, help="Interface-centered coordinate x-(center-1)")
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
            x_abs = (center - 1) + args.x_rel
            if x_abs < 0 or x_abs >= meta["N"]:
                raise SystemExit(f"x-rel={args.x_rel} is outside valid range for N={meta['N']}.")
            q_val, j_val = compute_point(C_t, x_abs, args.r, args.sign)
            zeta_val = float(x_abs - (center - 1)) / float(meta["time"])
            rows.append((meta["time"], zeta_val, q_val, j_val, meta))

    rows.sort(key=lambda item: item[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    zeta_axis = np.array([r[1] for r in rows], dtype=float)
    q_axis = np.array([r[2] for r in rows], dtype=float)
    j_axis = np.array([r[3] for r in rows], dtype=float)
    meta0 = rows[0][4]

    os.makedirs(args.outdir, exist_ok=True)
    base = base_name(meta0, args.x_rel, args.r, args.sign)
    out_csv = os.path.join(args.outdir, base + ".csv")
    out_png = os.path.join(args.outdir, base + ".png")

    np.savetxt(
        out_csv,
        np.column_stack([t_axis, zeta_axis, q_axis, j_axis]),
        delimiter=",",
        header="time,zeta_at_x,q,j",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    axes[0].plot(t_axis, q_axis, color="tab:blue", lw=1.8)
    axes[1].plot(t_axis, j_axis, color="tab:red", lw=1.8)
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$q(x,t)$")
    axes[1].set_ylabel(r"$J(x,t)$")
    fig.suptitle(rf"$x={args.x_rel},\ r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={meta0['gamma']},\ N={meta0['N']}$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {len(rows)} time slices")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
