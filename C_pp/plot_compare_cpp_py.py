#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_name(name, pattern):
    m = pattern.match(name)
    if not m:
        return None
    d = m.groupdict()
    info = {
        "r": int(d["r"]),
        "sign": d["sign"],
        "gamma": float(d["gamma"]),
        "N": int(d["N"]),
    }
    if "L" in d:
        info["L"] = float(d["L"])
    if "T" in d:
        info["T"] = float(d["T"])
    return info


def find_match_py(csv_dir, r, sign, gamma, N, pattern, tol=1e-6):
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_name(fname, pattern)
        if info is None:
            continue
        if info["r"] != r:
            continue
        if info["sign"] != sign:
            continue
        if not np.isclose(info["gamma"], gamma, atol=tol, rtol=0):
            continue
        if info["N"] != N:
            continue
        return os.path.join(csv_dir, fname)
    return None


def find_match_cpp(csv_dir, r, sign, gamma, N, pattern, T=None, L=None, tol=1e-6):
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_name(fname, pattern)
        if info is None:
            continue
        if info["r"] != r:
            continue
        if info["sign"] != sign:
            continue
        if not np.isclose(info["gamma"], gamma, atol=tol, rtol=0):
            continue
        if info["N"] != N:
            continue
        if T is not None and not np.isclose(info.get("T", np.nan), T, atol=tol, rtol=0):
            continue
        if L is not None and not np.isclose(info.get("L", np.nan), L, atol=tol, rtol=0):
            continue
        return os.path.join(csv_dir, fname)
    return None


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    zeta = data[:, 2]
    q_vals = data[:, 3]
    j_vals = data[:, 4]
    return zeta, q_vals, j_vals


def main():
    ap = argparse.ArgumentParser(description="Side-by-side comparison: Python vs C++ CSV outputs.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory.")
    ap.add_argument("--r", type=int, required=True, help="r value.")
    ap.add_argument("--sign", type=str, required=True, help="sign (+ or -).")
    ap.add_argument("--gamma", type=float, required=True, help="gamma value.")
    ap.add_argument("--time", type=float, default=None, help="T value (optional; used for C++ match).")
    ap.add_argument("--N", type=int, required=True, help="Total system size N.")
    ap.add_argument("--L", type=float, default=None, help="Monitored length L (optional; used for C++ match).")
    ap.add_argument("--zmin", type=float, default=-2.5, help="Min zeta for x-axis.")
    ap.add_argument("--zmax", type=float, default=2.5, help="Max zeta for x-axis.")
    ap.add_argument("--outpng", type=str, default=None, help="Output PNG path.")
    args = ap.parse_args()

    py_dir = os.path.join(args.outdir, "GHD_NEEL_PARTIAL_CSV")
    cpp_dir = os.path.join(args.outdir, "GHD_NEEL_PARTIAL_CSV_CPP")
    if not os.path.isdir(py_dir):
        raise SystemExit(f"Missing Python CSV dir: {py_dir}")
    if not os.path.isdir(cpp_dir):
        raise SystemExit(f"Missing C++ CSV dir: {cpp_dir}")

    pattern_py = re.compile(
        r"GHD_VCNEEL_l(?P<L>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_cpp = re.compile(
        r"GHD_VCNEEL_l(?P<L>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )

    py_path = find_match_cpp(py_dir, args.r, args.sign, args.gamma, args.N, pattern_py, T=args.time, L=args.L)
    cpp_path = find_match_cpp(cpp_dir, args.r, args.sign, args.gamma, args.N, pattern_cpp, T=args.time, L=args.L)
    if not py_path:
        raise SystemExit("No matching Python CSV found for the given parameters.")
    if not cpp_path:
        raise SystemExit("No matching C++ CSV found for the given parameters.")

    z_py, q_py, j_py = load_csv(py_path)
    z_cpp, q_cpp, j_cpp = load_csv(cpp_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(z_py, q_py, lw=1.6, color="blue", label="py q")
    axes[1].plot(z_py, j_py, lw=1.6, color="red", label="py J")
    axes[0].plot(z_cpp, q_cpp, lw=1.6, color="green", ls="--", label="cpp q")
    axes[1].plot(z_cpp, j_cpp, lw=1.6, color="orange", ls="--", label="cpp J")

    for ax in axes:
        ax.set_xlabel(r"$\zeta$")
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_xlim(args.zmin, args.zmax)
        ax.legend()

    axes[0].set_ylabel("q")
    axes[1].set_ylabel("J")

    title = rf"$r={args.r},\ \mathrm{{sign}}={args.sign},\ \gamma={args.gamma},\ N={args.N}$"
    if args.time is not None:
        title += rf",\ T={args.time}"
    if args.L is not None:
        title += rf",\ L={args.L}"
    fig.suptitle(title)
    fig.tight_layout()

    if args.outpng is None:
        png_dir = os.path.join(args.outdir, "GHD_NEEL_PARTIAL_PNG_CPP_PY")
        os.makedirs(png_dir, exist_ok=True)
        outpng = os.path.join(
            png_dir,
            f"cmp_cpp_py_r{args.r}_sign{args.sign}_gamma{args.gamma:.2f}_N{args.N}.png",
        )
    else:
        outpng = args.outpng

    fig.savefig(outpng, dpi=200)
    plt.close(fig)
    print(f"wrote {outpng}")


if __name__ == "__main__":
    main()
