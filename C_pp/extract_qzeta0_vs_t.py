#!/usr/bin/env python3
"""
Extract q and J at zeta=0, together with the immediate left/right neighbors,
from GHD_IT_CSV profile files.

This assumes the profile CSVs use the current left-interface convention:
    zeta = (x - (center - 1)) / T
so zeta=0 is the last site of the left region.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_qx_vs_t import parse_it_name


INIT_STATES = ("neel", "neel_even", "beta", "beta_lr", "vac_fill", "vac_infty", "mixed_neel", "phsymm", "phsymm_odd")


def build_base_name(info, num_pairs):
    prefix = {
        "neel": "GHD_QZ0_NEEL",
        "neel_even": "GHD_QZ0_NEEL_EVEN",
        "beta": "GHD_QZ0_Beta",
        "beta_lr": "GHD_QZ0_betaLR",
        "vac_fill": "GHD_QZ0_VAC_FILL",
        "vac_infty": "GHD_QZ0_VAC_INFTY",
        "mixed_neel": "GHD_QZ0_MIXED_NEEL",
        "phsymm": "GHD_QZ0_PHSYMM",
        "phsymm_odd": "GHD_QZ0_PHSYMM_ODD",
    }[info["kind"]]
    base = (
        f"{prefix}_r{info['r']}_s_{info['s']}_sign{info['sign']}"
        f"_gamma{info['gamma']:.2f}_N{info['N']}"
    )
    if num_pairs != 1:
        base += f"_pairs{num_pairs}"
    return base


def infer_x_abs(row, info):
    center = info["N"] // 2
    return int(round(float(row[2]) * float(info["T"]) + (center - 1)))


def main():
    ap = argparse.ArgumentParser(description="Extract q and J at zeta=0 from GHD_IT_CSV files.")
    ap.add_argument("--csv-dir", required=True, help="Directory containing GHD_IT CSV files")
    ap.add_argument("--outdir", default=".", help="Directory for extracted CSV/PNG")
    ap.add_argument("--init-state", required=True, choices=INIT_STATES)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sign", required=True, choices=["+", "-"])
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--size", type=int, required=True, help="Half-size L, so N=2L")
    ap.add_argument("--s-offset", type=int, required=True)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--phsymm-m", type=int, default=None)
    ap.add_argument("--phsymm-A", type=float, default=None)
    ap.add_argument("--time-min", type=float, default=None)
    ap.add_argument("--time-max", type=float, default=None)
    ap.add_argument("--num-pairs", type=int, default=1, help="Number of symmetric left/right neighbor pairs around zeta=0.")
    args = ap.parse_args()
    if args.num_pairs < 1:
        raise SystemExit("--num-pairs must be at least 1.")

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
        if data.ndim == 1:
            data = data[None, :]
        zeta = data[:, 2]
        zero_matches = np.where(np.isclose(zeta, 0.0, atol=1e-12))[0]
        if zero_matches.size == 1:
            idx = int(zero_matches[0])
        elif zero_matches.size == 0:
            idx = int(np.argmin(np.abs(zeta)))
            print(f"warning: no exact zeta=0 in {fname}; using nearest zeta={zeta[idx]:.12g}")
        else:
            raise SystemExit(f"Found multiple zeta=0 rows in {fname}")

        if idx < args.num_pairs or idx > (data.shape[0] - 1 - args.num_pairs):
            raise SystemExit(
                f"zeta=0 in {fname} is too close to a boundary for --num-pairs={args.num_pairs}"
            )

        def row_to_tuple(row):
            if row.shape[0] >= 6:
                x_abs = int(round(row[5]))
            else:
                x_abs = infer_x_abs(row, info)
            return (x_abs, float(row[2]), float(row[3]), float(row[4]))

        rows.append(
            (
                float(info["T"]),
                [row_to_tuple(data[idx + offset]) for offset in range(-args.num_pairs, args.num_pairs + 1)],
                info,
            )
        )

    if not rows:
        raise SystemExit("No matching CSV files found.")

    rows.sort(key=lambda item: item[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    offsets = list(range(-args.num_pairs, args.num_pairs + 1))
    block_arrays = {}
    for block_idx, offset in enumerate(offsets):
        block_arrays[offset] = {
            "x": np.array([r[1][block_idx][0] for r in rows], dtype=int),
            "zeta": np.array([r[1][block_idx][1] for r in rows], dtype=float),
            "q": np.array([r[1][block_idx][2] for r in rows], dtype=float),
            "j": np.array([r[1][block_idx][3] for r in rows], dtype=float),
        }

    info0 = rows[0][2]

    base = build_base_name(info0, args.num_pairs)
    out_csv = os.path.join(args.outdir, base + ".csv")
    out_png = os.path.join(args.outdir, base + ".png")

    csv_cols = [t_axis]
    csv_header = ["time"]
    for offset in offsets:
        tag = "x0" if offset == 0 else f"m{abs(offset)}" if offset < 0 else f"p{offset}"
        csv_cols.extend(
            [
                block_arrays[offset]["x"],
                block_arrays[offset]["zeta"],
                block_arrays[offset]["q"],
                block_arrays[offset]["j"],
            ]
        )
        csv_header.extend([f"x_{tag}", f"zeta_{tag}", f"q_{tag}", f"j_{tag}"])

    np.savetxt(
        out_csv,
        np.column_stack(csv_cols),
        delimiter=",",
        header=",".join(csv_header),
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("coolwarm")
    for i, offset in enumerate(offsets):
        frac = 0.5 if len(offsets) == 1 else i / (len(offsets) - 1)
        color = cmap(frac)
        x0 = block_arrays[offset]["x"][0]
        q0 = block_arrays[offset]["q"]
        if offset == 0:
            label = rf"$\zeta=0$ ($x={x0}$)"
            size = 30
        elif offset < 0:
            label = rf"$x_0-{abs(offset)}$ ($x={x0}$)"
            size = 22
        else:
            label = rf"$x_0+{offset}$ ($x={x0}$)"
            size = 22
        ax.scatter(t_axis, q0, color=color, s=size, label=label)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$q(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2 if args.num_pairs > 2 else 1, fontsize=9)
    ax.set_title(
        rf"$r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={args.gamma},\ N={2*args.size}$"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {len(rows)} time slices")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
