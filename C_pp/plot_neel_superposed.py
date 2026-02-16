#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

'''
run example:
python3 plot_neel_superposed.py \
  --outdir /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp \
  --r-list "0 2" \
  --sign "+" \
  --gammas "0.5 1.0" \
  --sizes "1000" \
  --beta "1.0"
  --time 200 \
  --a-offset 1000 \
  --b-offsets "1 5 10 50 100 500 1000"


'''
def parse_list(s, cast=float):
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    return [cast(p) for p in parts]


def parse_name(path):
    name = os.path.basename(path)
    pattern_neel = re.compile(
        r"GHD_NEEL_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta = re.compile(
        r"GHD_NEEL_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta_lr = re.compile(
        r"GHD_NEEL_betaL_(?P<betaL>[-0-9.]+)_betaR_(?P<betaR>[-0-9.]+)_r(?P<r>[-0-9]+)_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    m = pattern_neel.match(name)
    if m:
        d = m.groupdict()
        return {
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "a": int(d["a"]),
            "b": int(d["b"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
    m = pattern_beta_lr.match(name)
    if m:
        d = m.groupdict()
        return {
            "beta": None,
            "betaL": float(d["betaL"]),
            "betaR": float(d["betaR"]),
            "a": int(d["a"]),
            "b": int(d["b"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
    m = pattern_beta.match(name)
    if m:
        d = m.groupdict()
        return {
            "beta": float(d["beta"]),
            "betaL": None,
            "betaR": None,
            "a": int(d["a"]),
            "b": int(d["b"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
    return None


def main():
    ap = argparse.ArgumentParser(description="Overlay q/J profiles for each (r,gamma) with multiple B_OFFSET values.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory (CSV in GHD_SIMP_NELL_CSV).")
    ap.add_argument("--r-list", type=str, default=None, help="Comma/space-separated r values.")
    ap.add_argument("--sign", type=str, default=None, help="Filter by sign (+ or -).")
    ap.add_argument("--gammas", type=str, default=None, help="Comma/space-separated gamma values.")
    ap.add_argument("--beta", type=float, default=None, help="Filter by homogeneous Beta")
    ap.add_argument("--betaL", type=float, default=None, help="Filter by betaL")
    ap.add_argument("--betaR", type=float, default=None, help="Filter by betaR")
    ap.add_argument("--sizes", type=str, default=None, help="Comma/space-separated half-sizes (L).")
    ap.add_argument("--time", type=float, default=None, help="Filter by T.")
    ap.add_argument("--a-offset", type=int, default=None, help="Filter by A_OFFSET (left length).")
    ap.add_argument("--b-offsets", type=str, default=None, help="Comma/space-separated B_OFFSET values.")
    args = ap.parse_args()

    csv_dir = os.path.join(args.outdir, "GHD_SIMP_NELL_CSV")
    if not os.path.isdir(csv_dir):
        raise SystemExit(f"Missing CSV directory: {csv_dir}")

    r_list = parse_list(args.r_list, int)
    gamma_list = parse_list(args.gammas, float)
    size_list = parse_list(args.sizes, int)
    b_offset_list = parse_list(args.b_offsets, int)
    n_list = [2 * s for s in size_list] if size_list else None
    Beta = args.beta
    betaL = args.betaL
    betaR = args.betaR

    entries = []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_name(fname)
        if info is None:
            continue
        if Beta is not None and not np.isclose(info["beta"], Beta):
            continue
        if betaL is not None:
            if "betaL" not in info or not np.isclose(info["betaL"], betaL):
                continue
        if betaR is not None:
            if "betaR" not in info or not np.isclose(info["betaR"], betaR):
                continue
        if r_list is not None and info["r"] not in r_list:
            continue
        if args.sign is not None and info["sign"] != args.sign:
            continue
        if gamma_list is not None and not any(np.isclose(info["gamma"], g) for g in gamma_list):
            continue
        if args.time is not None and not np.isclose(info["T"], args.time):
            continue
        if n_list is not None and info["N"] not in n_list:
            continue
        if args.a_offset is not None and info["a"] is not None and info["a"] != args.a_offset:
            continue
        if b_offset_list is not None and info["b"] is not None and info["b"] not in b_offset_list:
            continue
        entries.append((os.path.join(csv_dir, fname), info))

    if not entries:
        raise SystemExit("No matching CSV files found.")

    # group by (r, gamma)
    groups = {}
    for path, info in entries:
        key = (info["beta"], info["r"], info["gamma"])
        groups.setdefault(key, []).append((path, info))

    png_dir = os.path.join(args.outdir, "GHD_SIMP_NELL_PNG")
    os.makedirs(png_dir, exist_ok=True)

    for (beta_val, r_val, gamma_val), items in groups.items():
        items.sort(key=lambda x: x[1]["b"])
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        markers = ["o", "s", "^", "D", "v", "x", "+", "*", "P", "X"]
        for idx, (path, info) in enumerate(items):
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            zeta = data[:, 2]
            q_vals = data[:, 3]
            j_vals = data[:, 4]
            color = cmap(idx / max(len(items) - 1, 1))
            label = f"B_OFFSET={int(info['b'])}"
            marker = markers[idx % len(markers)]
            axes[0].scatter(zeta, q_vals, s=6, marker=marker, color=color, alpha=0.8, label=label)
            axes[1].scatter(zeta, j_vals, s=6, marker=marker, color=color, alpha=0.8, label=label)

        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8)
            ax.set_xlim(-2.5, 2.5)
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")
        

        info = items[0][1]
        if beta_val == 0:
            title = rf"$r={info['r']},\ \mathrm{{sign}}={info['sign']},\ \gamma={info['gamma']},\ T={info['T']},\ N={info['N']},\ A={info['a']}$"
        elif beta_val is None and info.get("betaL") is not None and info.get("betaR") is not None:
            title = rf"$\beta_L={info['betaL']},\ \beta_R={info['betaR']},\ r={info['r']},\ \mathrm{{sign}}={info['sign']},\ \gamma={info['gamma']},\ T={info['T']},\ N={info['N']},\ A={info['a']}$"
        else:
            title = rf"$\beta={beta_val},\ r={info['r']},\ \mathrm{{sign}}={info['sign']},\ \gamma={info['gamma']},\ T={info['T']},\ N={info['N']},\ A={info['a']}$"
        fig.suptitle(title)
        fig.tight_layout()
        if beta_val == 0:
            outpng = os.path.join(png_dir, f"GHD_NEEL_superposed_r{r_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}.png")
        elif beta_val is None and info.get("betaL") is not None and info.get("betaR") is not None:
            outpng = os.path.join(png_dir, f"GHD_NEEL_superposed_betaL{info['betaL']}_betaR{info['betaR']}_r{r_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}.png")            
        else:
            outpng = os.path.join(png_dir, f"GHD_NEEL_superposed_Beta{beta_val}_r{r_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}.png")            
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outpng}")


if __name__ == "__main__":
    main()
