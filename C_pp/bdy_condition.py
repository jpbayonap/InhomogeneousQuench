#!/usr/bin/env python3
"""
Extract boundary-condition diagnostics at a fixed interface-centered coordinate x_rel
directly from covariance files produced by main_dynamics_it_cov.py.

run example:
cd /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp
source ~/venvs/qsim/bin/activate

python3 bdy_condition.py \
  --outdir ./runs/neel_cov_local/GHD_BDY \
  --cov-dir ./runs/neel_cov_local/GHD_IT_COV \
  --init-state neel \
  --sizes "1000" \
  --s-offsets "1000" \
  --gammas "0.5 1.0" \
  --times "100 150 200 250 300 400" \
  --r 3 \
  --delta 5 \
  --sign - \
  --x-rel 0

python3 bdy_condition.py \
  --outdir ./runs/beta_cov_local_L1500/GHD_BDY \
  --cov-dir ./runs/beta_cov_local_L1500/GHD_IT_COV \
  --init-state beta \
  --sizes "1500" \
  --s-offsets "1500" \
  --gammas "1.0" \
  --times "150 200 250 300 350 400" \
  --beta 1.0 \
  --r 4 \
  --delta 5 \
  --sign + \
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

from profiles_from_cov_it import (
    INIT_STATES,
    build_cov_search_dir,
    jm_left,
    jp_left,
    load_meta,
    matches,
    parse_list,
    qp_left,
    qm_left,
)


def compute_point(C_t, delta, x_center, r, gamma, sign, state):
    if sign == "-":
        q_fun = qm_left
        j_fun = jm_left
    elif sign == "+":
        q_fun = qp_left
        j_fun = jp_left
    else:
        raise ValueError(f"Unsupported sign: {sign}")

    x_array = np.arange(-r + 1, 1) + x_center
    q_center = float(np.sum([np.real(q_fun(r, x, C_t, "open")) for x in x_array]))
    j_r = float(np.real(j_fun(r, x_center + delta, C_t, "open")))
    j_l = float(np.real(j_fun(r, x_center - delta, C_t, "open")))
    q_l = float(np.real(q_fun(r, x_center - delta, C_t, "open")))
    q_r = float(np.real(q_fun(r, x_center + delta, C_t, "open")))
    Delta_j = j_r - j_l
    rhs_term = gamma * q_center
    prefactor = 1.0 if state == "beta" else -r
    q_l = prefactor * gamma * q_l
    q_r = prefactor * gamma * q_r
    error_bdy = np.abs(Delta_j + rhs_term)
    error_l = np.abs(Delta_j + q_l)
    error_r = np.abs(Delta_j + q_r)
    return Delta_j, rhs_term, q_l, q_r, error_bdy, error_l, error_r


def base_name(meta0, x_rel, delta, r, sign):
    prefix = {
        "neel": "GHD_BDY_NEEL_COV",
        "neel_even": "GHD_BDY_NEEL_EVEN_COV",
        "beta": "GHD_BDY_Beta_COV",
        "beta_lr": "GHD_BDY_betaLR_COV",
        "vac_fill": "GHD_BDY_VAC_FILL_COV",
        "mixed_neel": "GHD_BDY_MIXED_NEEL_COV",
        "vac_infty": "GHD_BDY_VAC_INFTY_COV",
        "phsymm": "GHD_BDY_PHSYMM_COV",
        "phsymm_odd": "GHD_BDY_PHSYMM_ODD_COV",
    }[meta0["init_state"]]
    pieces = [
        prefix,
        f"x{x_rel}",
        f"delta{delta}",
        f"r{r}",
        f"s_{meta0['s_offset']}",
        f"sign{sign}",
    ]
    if meta0["init_state"] == "beta":
        pieces.append(f"beta{meta0['beta']:.6g}")
    elif meta0["init_state"] == "beta_lr":
        pieces.append(f"betaL{meta0['betaL']:.6g}")
        pieces.append(f"betaR{meta0['betaR']:.6g}")
    elif meta0["init_state"] in {"phsymm", "phsymm_odd"}:
        pieces.append(f"m{meta0['phsymm_m']}")
        pieces.append(f"A{meta0['phsymm_A']:.6g}")
    pieces.append(f"gamma{meta0['gamma']:.2f}")
    pieces.append(f"N{meta0['N']}")
    return "_".join(pieces)


def main():
    ap = argparse.ArgumentParser(description="Extract q(x,t), J(x,t) directly from GHD_IT_COV files.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for CSV/PNG")
    ap.add_argument("--cov-dir", type=str, default=None, help="Directory with covariance .npz files")
    ap.add_argument("--init-state", type=str, required=True, choices=INIT_STATES)
    ap.add_argument("--sizes", type=str, required=True, help="Comma/space-separated half-sizes L")
    ap.add_argument("--s-offsets", type=str, required=True, help="Comma/space-separated s offsets")
    ap.add_argument("--gammas", type=str, required=True, help="Comma/space-separated gamma values")
    ap.add_argument("--times", type=str, default=None, help="Optional comma/space-separated times")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--sign", type=str, required=True, choices=["+", "-"])
    ap.add_argument("--x-rel", type=int, default=0, help="Interface-centered coordinate x-(center-1)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--phsymm-m", type=int, default=None)
    ap.add_argument("--phsymm-A", type=float, default=None)

    args = ap.parse_args()

    if args.init_state in {"beta", "beta_lr"}:
        if args.sign != "+":
            raise SystemExit(f"--init-state {args.init_state} requires --sign +.")
        if args.r % 2 == 0:
            raise SystemExit(f"--init-state {args.init_state} requires odd --r.")

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
            delta_j, rhs_term, q_l, q_r, error_bdy, error_l, error_r = compute_point(
                C_t, args.delta, x_center, args.r, meta["gamma"], args.sign, meta["init_state"]
            )
            zeta_val = float(x_center - (center - 1)) / float(meta["time"])
            rows.append((meta["time"], zeta_val, delta_j, rhs_term, error_bdy, q_l, error_l, q_r, error_r, meta))


    # Sort by time rows[0]
    rows.sort(key=lambda item: item[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    zeta_axis = np.array([r[1] for r in rows], dtype=float)
    delta_j = np.array([r[2] for r in rows], dtype=float)
    rhs_term = np.array([r[3] for r in rows], dtype=float)
    error_bdy = np.array([r[4] for r in rows], dtype=float)
    q_l = np.array([r[5] for r in rows], dtype=float)
    error_l = np.array([r[6] for r in rows], dtype=float)
    q_r = np.array([r[7] for r in rows], dtype=float)
    error_r = np.array([r[8] for r in rows], dtype=float)
    # Metadata dictionary from the first row after sorting.
    meta0 = rows[0][9]
    x_center0 = (meta0["center"] - 1) + args.x_rel


    os.makedirs(args.outdir, exist_ok=True)
    base = base_name(meta0, args.x_rel, args.delta, args.r, args.sign)
    out_csv = os.path.join(args.outdir, base + ".csv")
    out_png = os.path.join(args.outdir, base + ".png")

    np.savetxt(
        out_csv,
        np.column_stack([t_axis, zeta_axis, delta_j, rhs_term,  error_bdy, q_l, error_l, q_r, error_r]),
        delimiter=",",
        header="time,zeta_at_x,delta_j,rhs_term,error_bdy,q_l,error_l,q_r,error_r",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    sign_tex = args.sign
    edge_factor_tex = r"\gamma" if meta0["init_state"] == "beta" else r"r\gamma"
    axes[0].scatter(
        t_axis,
        error_bdy,
        color="tab:blue",
        label=rf"$|\Delta J^{{(r,{sign_tex})}} + \gamma \sum_{{x=-r+1}}^0 q^{{(r,{sign_tex})}}_x|$",
        s=8.8,
    )
    axes[0].scatter(
        t_axis,
        error_l,
        color="tab:green",
        label=rf"$|\Delta J^{{(r,{sign_tex})}} + {edge_factor_tex} q^{{(r,{sign_tex})}}_{{-\delta}}|$",
        s=8.8,
    )
    axes[0].scatter(
        t_axis,
        error_r,
        color="tab:orange",
        label=rf"$|\Delta J^{{(r,{sign_tex})}} + {edge_factor_tex} q^{{(r,{sign_tex})}}_{{+\delta}}|$",
        s=8.8,
    )
    axes[1].scatter(t_axis, abs(delta_j), color="tab:green", s=8.8, marker="s", label=rf"$|\Delta J^{{(r,{sign_tex})}}|$")
    axes[1].scatter(
        t_axis,
        abs(rhs_term),
        color="tab:red",
        s=8.8,
        marker="^",
        label=rf"$|\gamma \sum_{{x=-r+1}}^0 q^{{(r,{sign_tex})}}_x|$",
    )
    axes[1].scatter(t_axis, abs(q_l), color="tab:blue", s=8.8, marker="x", label=rf"$|{edge_factor_tex} q^{{(r,{sign_tex})}}_{{-\delta}}|$")
    axes[1].scatter(t_axis, abs(q_r), color="tab:orange", s=8.8, marker="o", label=rf"$|{edge_factor_tex} q^{{(r,{sign_tex})}}_{{+\delta}}|$")
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.grid(True, alpha=0.25)
    
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].legend()

    fig.suptitle(rf"$x_0={x_center0},\ \delta={args.delta},\ r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={meta0['gamma']},\ N={meta0['N']}$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {len(rows)} time slices")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
