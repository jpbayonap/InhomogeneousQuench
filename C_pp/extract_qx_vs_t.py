#!/usr/bin/env python3
"""
Extract q(x,t) at a fixed interface-centered coordinate x_rel from GHD_IT_CSV files.

Convention:
    x_rel = x_abs - (center - 1),  center = N // 2
so x_rel=0 means the last site of the left region and zeta = 0 there.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_it_name(name):
    patterns = [
        (
            re.compile(
                r"GHD_IT_NEEL_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "neel",
        ),
        (
            re.compile(
                r"GHD_IT_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "beta",
        ),
        (
            re.compile(
                r"GHD_IT_betaL_(?P<betaL>[-0-9.]+)_betaR_(?P<betaR>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "beta_lr",
        ),
        (
            re.compile(
                r"GHD_vac_fill_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "vac_fill",
        ),
        (
            re.compile(
                r"GHD_IT_VacInfty_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "vac_infty",
        ),
        (
            re.compile(
                r"GHD_IT_MixedNeel_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "mixed_neel",
        ),
        (
            re.compile(
                r"GHD_IT_PHSYMM_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "phsymm",
        ),
        (
            re.compile(
                r"GHD_IT_PHSYMM_ODD_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv$"
            ),
            "phsymm_odd",
        ),
    ]

    for pat, kind in patterns:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        out = {
            "kind": kind,
            "r": int(d["r"]),
            "s": int(float(d["s"])),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "m": int(d["m"]) if "m" in d and d["m"] is not None else None,
            "A": float(d["A"]) if "A" in d and d["A"] is not None else None,
        }
        if kind == "beta":
            out["beta"] = float(d["beta"])
        elif kind == "beta_lr":
            out["betaL"] = float(d["betaL"])
            out["betaR"] = float(d["betaR"])
        else:
            out["beta"] = 0.0 if kind == "neel" else None
        return out
    return None


def build_base_name(info, x_rel):
    prefix = {
        "neel": "GHD_QXT_NEEL",
        "beta": "GHD_QXT_Beta",
        "beta_lr": "GHD_QXT_betaLR",
        "vac_fill": "GHD_QXT_VAC_FILL",
        "vac_infty": "GHD_QXT_VAC_INFTY",
        "mixed_neel": "GHD_QXT_MIXED_NEEL",
        "phsymm": "GHD_QXT_PHSYMM",
        "phsymm_odd": "GHD_QXT_PHSYMM_ODD",
    }[info["kind"]]
    return (
        f"{prefix}_x{x_rel}_r{info['r']}_s_{info['s']}_sign{info['sign']}"
        f"_gamma{info['gamma']:.2f}_N{info['N']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Extract q(x,t) from GHD_IT_CSV files at a fixed interface-centered x.")
    ap.add_argument("--csv-dir", required=True, help="Directory containing GHD_IT CSV files")
    ap.add_argument("--outdir", default=".", help="Directory for extracted CSV/PNG")
    ap.add_argument("--init-state", required=True, choices=["neel", "beta", "beta_lr", "vac_fill", "vac_infty", "mixed_neel", "phsymm", "phsymm_odd"])
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sign", required=True, choices=["+", "-"])
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--size", type=int, required=True, help="Half-size L, so N=2L")
    ap.add_argument("--s-offset", type=int, required=True)
    ap.add_argument("--x-rel", type=int, default=0, help="Interface-centered coordinate x-(center-1)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--phsymm-m", type=int, default=None)
    ap.add_argument("--phsymm-A", type=float, default=None)
    ap.add_argument("--time-min", type=float, default=None)
    ap.add_argument("--time-max", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for fname in os.listdir(args.csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_it_name(fname)
        if info is None:
            continue
        if info["kind"] != args.init_state:
            continue
        if info["r"] != args.r or info["sign"] != args.sign:
            continue
        if not np.isclose(info["gamma"], args.gamma):
            continue
        if info["N"] != 2 * args.size:
            continue
        if info["s"] != args.s_offset:
            continue
        if args.time_min is not None and info["T"] < args.time_min:
            continue
        if args.time_max is not None and info["T"] > args.time_max:
            continue
        if args.init_state == "beta" and args.beta is not None and not np.isclose(info.get("beta"), args.beta):
            continue
        if args.init_state == "beta_lr":
            if args.betaL is not None and not np.isclose(info.get("betaL"), args.betaL):
                continue
            if args.betaR is not None and not np.isclose(info.get("betaR"), args.betaR):
                continue
        if args.init_state in ("phsymm", "phsymm_odd"):
            if args.phsymm_m is not None and info.get("m") != args.phsymm_m:
                continue
            if args.phsymm_A is not None and not np.isclose(info.get("A"), args.phsymm_A):
                continue

        path = os.path.join(args.csv_dir, fname)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        n_rows = data.shape[0]
        center = info["N"] // 2
        x_abs = (center - 1) + args.x_rel
        if x_abs < 0 or x_abs >= n_rows:
            raise SystemExit(f"x-rel={args.x_rel} is outside valid range for N={info['N']}.")
        row = data[x_abs]
        rows.append((float(info["T"]), float(row[2]), float(row[3]), float(row[4]), info))

    if not rows:
        raise SystemExit("No matching CSV files found.")

    rows.sort(key=lambda item: item[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    zeta_axis = np.array([r[1] for r in rows], dtype=float)
    q_axis = np.array([r[2] for r in rows], dtype=float)
    j_axis = np.array([r[3] for r in rows], dtype=float)
    info0 = rows[0][4]

    base = build_base_name(info0, args.x_rel)
    out_csv = os.path.join(args.outdir, base + ".csv")
    out_png = os.path.join(args.outdir, base + ".png")

    np.savetxt(
        out_csv,
        np.column_stack([t_axis, zeta_axis, q_axis, j_axis]),
        delimiter=",",
        header="time,zeta_at_x,q,j",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(t_axis, q_axis, color="tab:blue", lw=1.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$q(x,t)$")
    ax.grid(True, alpha=0.25)
    ax.set_title(rf"$x={args.x_rel},\ r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={args.gamma},\ N={2*args.size}$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {len(rows)} time slices")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
