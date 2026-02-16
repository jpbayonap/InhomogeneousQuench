#!/usr/bin/env python3
"""
Plot numerics vs GHD thermal predictions using saved CSVs.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().strip('"') for h in header]
        if len(header) < 2 or header[0] == "-Transpose-":
            return None
        cols = [[] for _ in header]
        for row in reader:
            if not row:
                continue
            for i, val in enumerate(row):
                cols[i].append(float(val.strip().strip('"')))
    data = np.zeros(len(cols[0]), dtype=[(h, float) for h in header])
    for h, col in zip(header, cols):
        data[h] = col
    return data


def is_valid_mat_csv(path):
    try:
        if os.path.getsize(path) < 20:
            return False
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            return False
        header = [h.strip().strip('"') for h in header]
        if len(header) < 2 or header[0] == "-Transpose-":
            return False
        return True
    except OSError:
        return False


def parse_meta(filename):
    base = os.path.basename(filename)
    meta = {}
    m = re.search(r"betaL_([0-9.]+)betaR_([0-9.]+)", base)
    if m:
        meta["betaL"] = float(m.group(1))
        meta["betaR"] = float(m.group(2))
    m = re.search(r"_r(-?\d+)_sign([+-])", base)
    if m:
        meta["r"] = int(m.group(1))
        meta["sign"] = m.group(2)
    m = re.search(r"_gamma([0-9.]+)", base)
    if m:
        meta["gamma"] = float(m.group(1))
    return meta


def parse_mat_meta(filename):
    base = os.path.basename(filename)
    meta = {}
    m = re.search(r"_r(-?\d+)_sign([+-])", base)
    if m:
        meta["r"] = int(m.group(1))
        meta["sign"] = m.group(2)
    m = re.search(r"_M(\d+)", base)
    if m:
        meta["M"] = int(m.group(1))
    m = re.search(r"_gamma([0-9.]+)", base)
    if m:
        meta["gamma"] = float(m.group(1).rstrip("."))
    m = re.search(r"_beta([0-9.]+)", base)
    if m:
        meta["beta"] = float(m.group(1).rstrip("."))
    return meta


def build_mat_index(mat_dir):
    mat_index = {}
    mat_M = {}
    for path in glob.glob(os.path.join(mat_dir, "GHD_THERM_MAT_*.csv")):
        if not is_valid_mat_csv(path):
            continue
        meta = parse_mat_meta(path)
        key = (meta.get("r"), meta.get("sign"), meta.get("gamma"), meta.get("beta"))
        M = meta.get("M")
        if None in key:
            continue
        if M is None:
            mat_index[key] = path
            continue
        if key not in mat_M or M > mat_M[key]:
            mat_M[key] = M
            mat_index[key] = path
    return mat_index


def build_mat_index_with_M(mat_dir, target_M):
    mat_index = {}
    for path in glob.glob(os.path.join(mat_dir, "GHD_THERM_MAT_*.csv")):
        if not is_valid_mat_csv(path):
            continue
        meta = parse_mat_meta(path)
        key = (meta.get("r"), meta.get("sign"), meta.get("gamma"), meta.get("beta"))
        M = meta.get("M")
        if None in key or M is None:
            continue
        if M == target_M:
            mat_index[key] = path
    return mat_index


def plot_one(numerics_path, args, mat_index):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    data_num = load_csv(numerics_path)
    if data_num is None:
        raise ValueError(f"Invalid numerics CSV: {numerics_path}")
    names = data_num.dtype.names
    zeta = data_num["zeta"] if "zeta" in names else data_num[names[2]]
    q = data_num["q"] if "q" in names else data_num[names[3]]
    j = data_num["j"] if "j" in names else data_num[names[4]]

    meta = parse_meta(numerics_path)
    betaL = args.betaL if args.betaL is not None else meta.get("betaL")
    betaR = args.betaR if args.betaR is not None else meta.get("betaR")
    r = args.r if args.r is not None else meta.get("r")
    sign = args.sign if args.sign is not None else meta.get("sign")

    if betaL is None or betaR is None or r is None or sign is None:
        raise ValueError(
            "Missing betaL/betaR/r/sign. Pass via args or use filename pattern."
        )

    beta = betaL if betaL == betaR else None
    key = (r, sign, meta.get("gamma"), beta)
    mat_path = mat_index.get(key)
    if not mat_path or not os.path.exists(mat_path):
        if args.verbose:
            print(
                "Missing MAT CSV for "
                f"r={r}, sign={sign}, gamma={meta.get('gamma')}, beta={beta}"
            )
        return
    data_mat = load_csv(mat_path)
    if data_mat is None:
        if args.verbose:
            print(f"Invalid MAT CSV (no data): {mat_path}")
        return
    mat_names = data_mat.dtype.names
    z_ghd = data_mat["zeta"] if "zeta" in mat_names else data_mat[mat_names[0]]
    q_ghd = data_mat["q"] if "q" in mat_names else data_mat[mat_names[1]]
    j_ghd = data_mat["J"] if "J" in mat_names else data_mat[mat_names[2]]
    if args.stride > 1:
        z_ghd = z_ghd[::args.stride]
        q_ghd = q_ghd[::args.stride]
        j_ghd = j_ghd[::args.stride]

    if args.interp:
        zeta_scale = 1.0
        if args.zeta_scale_fit:
            fit_min = args.zeta_fit_min
            fit_max = args.zeta_fit_max
            span = args.zeta_scale_span
            scales = np.linspace(1.0 - span, 1.0 + span, 61)
            best_scale = 1.0
            best_rms = np.inf
            for s in scales:
                z_scaled = zeta * s
                q_scaled = np.interp(z_scaled, z_ghd, q_ghd)
                mask = (np.abs(zeta) >= fit_min) & (np.abs(zeta) <= fit_max)
                if not np.any(mask):
                    continue
                dq = q[mask] - q_scaled[mask]
                rms = np.sqrt(np.mean(dq * dq))
                if rms < best_rms:
                    best_rms = rms
                    best_scale = s
            zeta_scale = best_scale
            print(f"zeta scale fit: {zeta_scale:.6f} (span={span}, range={fit_min}..{fit_max})")
        z_plot = zeta
        q_ghd_plot = np.interp(zeta * zeta_scale, z_ghd, q_ghd)
        j_ghd_plot = np.interp(zeta * zeta_scale, z_ghd, j_ghd)
    else:
        q_ghd_plot = q_ghd
        j_ghd_plot = j_ghd
        z_plot = z_ghd

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(zeta, q, ".", ms=1, alpha=0.6, c="blue", label="numerics")
    axes[0].plot(z_plot, q_ghd_plot, "-", lw=2, c="black", label="GHD")
    axes[1].plot(zeta, j, ".", ms=1, alpha=0.6, c="red", label="numerics")
    axes[1].plot(z_plot, j_ghd_plot, "-", lw=2, c="orange", label="GHD")

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel(r"$\zeta$")
        ax.legend()

    axes[0].set_ylabel("q")
    axes[1].set_ylabel("J")

    title = f"betaL={betaL}, betaR={betaR}, r={r}, sign={sign}"
    if "gamma" in meta:
        title += f", gamma={meta['gamma']}"
    fig.suptitle(title)
    fig.tight_layout()

    outdir = args.png_dir or os.path.join(args.csv_dir, "..", "GHD_THERM_ALL_PNG")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, os.path.basename(numerics_path).replace(".csv", ".png"))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # Delta plots vs |zeta|.
    delta_q = q - q_ghd_plot
    delta_j = j - j_ghd_plot
    abs_z = np.abs(zeta)
    fit_mask = (abs_z >= 0.2) & (abs_z <= 1.8)
    fit_info = {}
    if np.any(fit_mask):
        z_fit = abs_z[fit_mask]
        q_fit = delta_q[fit_mask]
        j_fit = delta_j[fit_mask]
        mq, bq = np.polyfit(z_fit, q_fit, 1)
        mj, bj = np.polyfit(z_fit, j_fit, 1)
        fit_info = {"q": (mq, bq), "j": (mj, bj)}
    fig_d, ax_d = plt.subplots(1, 1, figsize=(6, 4))
    ax_d.plot(abs_z, delta_q, ".", ms=1, alpha=0.6, c="blue", label="delta_q")
    ax_d.plot(abs_z, delta_j, ".", ms=1, alpha=0.6, c="red", label="delta_J")
    if fit_info:
        z_line = np.linspace(0.2, 1.8, 200)
        mq, bq = fit_info["q"]
        mj, bj = fit_info["j"]
        ax_d.plot(z_line, mq * z_line + bq, c="blue", lw=2, alpha=0.8, label=f"fit q: {mq:+.3e}")
        ax_d.plot(z_line, mj * z_line + bj, c="red", lw=2, alpha=0.8, label=f"fit J: {mj:+.3e}")
    ax_d.set_xlabel(r"$|\zeta|$")
    ax_d.set_ylabel("delta")
    ax_d.grid(True, ls="--", alpha=0.4)
    ax_d.legend()
    ax_d.set_title(title)
    if fit_info:
        mq, bq = fit_info["q"]
        mj, bj = fit_info["j"]
        print(
            "delta vs |zeta| fit (0.2..1.8): "
            f"q slope={mq:+.3e}, intercept={bq:+.3e}; "
            f"J slope={mj:+.3e}, intercept={bj:+.3e}"
        )
    fig_d.tight_layout()
    out_delta = out.replace(".png", "_delta.png")
    fig_d.savefig(out_delta, dpi=200)
    plt.close(fig_d)
    if args.verbose:
        print(f"wrote {out_delta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="GHD_THERM_ALL_CSV")
    ap.add_argument("--mat-dir", default="GHD_THERM_MAT_CSV")
    ap.add_argument("--png-dir", default="GHD_THERM_COMP")
    ap.add_argument("--file", help="Plot a single numerics CSV file.")
    ap.add_argument("--betaL", type=float, default=None)
    ap.add_argument("--betaR", type=float, default=None)
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--sign", choices=["+", "-"], default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--interp", action="store_true", default=True)
    ap.add_argument("--no-interp", dest="interp", action="store_false")
    ap.add_argument("--zeta-scale-fit", action="store_true", help="Fit a small zeta rescale for GHD interpolation")
    ap.add_argument("--zeta-scale-span", type=float, default=0.03, help="Scale search half-span around 1.0")
    ap.add_argument("--zeta-fit-min", type=float, default=0.2, help="|zeta| fit min for scale")
    ap.add_argument("--zeta-fit-max", type=float, default=1.8, help="|zeta| fit max for scale")
    ap.add_argument("--M", type=int, default=None, help="Use MAT CSVs for this M")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.file:
        mat_index = (
            build_mat_index_with_M(args.mat_dir, args.M)
            if args.M is not None
            else build_mat_index(args.mat_dir)
        )
        plot_one(args.file, args, mat_index)
        return

    mat_index = (
        build_mat_index_with_M(args.mat_dir, args.M)
        if args.M is not None
        else build_mat_index(args.mat_dir)
    )
    pattern = os.path.join(
        args.csv_dir,
        "GHD_THERM_betaL_*betaR_*_r*_sign*_gamma*_N*_TEST.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return

    for path in files:
        plot_one(path, args, mat_index)


if __name__ == "__main__":
    main()
