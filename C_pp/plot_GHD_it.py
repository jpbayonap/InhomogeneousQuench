#!/usr/bin/env python3
import argparse
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is importable for ghd_module
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)

from ghd_module import hyd_charge_MICHELE, hyd_current_MICHELE


def parse_list(s, cast=float):
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    return [cast(p) for p in parts]


def parse_name(path):
    name = os.path.basename(path)
    pattern_neel = re.compile(
        r"GHD_IT_NEEL_r(?P<r>[-0-9.]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta = re.compile(
        r"GHD_IT_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta_lr = re.compile(
        r"GHD_IT_betaL_(?P<betaL>[-0-9.]+)_betaR_(?P<betaR>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )

    m = pattern_neel.match(name)
    if m:
        d = m.groupdict()
        return {
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "s": int(d["s"]),
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
            "s": int(d["s"]),
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
            "s": int(d["s"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }

    return None


def main():
    ap = argparse.ArgumentParser(description="Compare IT CSV profiles against GHD for fixed s-offset.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory (CSV in GHD_IT_CSV).")
    ap.add_argument("--r-list", type=str, default=None, help="Comma/space-separated r values.")
    ap.add_argument("--sign", type=str, default=None, help="Filter sign (+ or -); used for GHD if provided.")
    ap.add_argument("--gammas", type=str, default=None, help="Comma/space-separated gamma values.")
    ap.add_argument("--beta", type=float, default=None, help="Filter homogeneous Beta")
    ap.add_argument("--betaL", type=float, default=None, help="Filter betaL")
    ap.add_argument("--betaR", type=float, default=None, help="Filter betaR")
    ap.add_argument("--sizes", type=str, default=None, help="Comma/space-separated half-sizes (L).")
    ap.add_argument("--time", type=float, default=None, help="Filter by T.")
    ap.add_argument("--s-only", type=int, default=1, help="Keep only CSVs with this s-offset (default: 1).")
    args = ap.parse_args()

    csv_dir = os.path.join(args.outdir, "GHD_IT_CSV")
    if not os.path.isdir(csv_dir):
        raise SystemExit(f"Missing CSV directory: {csv_dir}")

    r_list = parse_list(args.r_list, int)
    gamma_list = parse_list(args.gammas, float)
    size_list = parse_list(args.sizes, int)
    n_list = [2 * s for s in size_list] if size_list else None

    entries = []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_name(fname)
        if info is None:
            continue

        # Force only the requested s-offset (default s=1)
        if info["s"] != args.s_only:
            continue
        if args.beta is not None and (info["beta"] is None or not np.isclose(info["beta"], args.beta)):
            continue
        if args.betaL is not None and (info.get("betaL") is None or not np.isclose(info["betaL"], args.betaL)):
            continue
        if args.betaR is not None and (info.get("betaR") is None or not np.isclose(info["betaR"], args.betaR)):
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
        entries.append((os.path.join(csv_dir, fname), info))

    if not entries:
        raise SystemExit(f"No matching CSV files found for s={args.s_only}.")

    groups = {}
    for path, info in entries:
        key = (info["beta"], info.get("betaL"), info.get("betaR"), info["r"], info["gamma"], info["T"], info["N"], info["sign"])
        groups.setdefault(key, []).append((path, info))

    png_dir = os.path.join(args.outdir, "GHD_IT_PNG_GHD")
    os.makedirs(png_dir, exist_ok=True)

    for (_, betaL_val, betaR_val, r_val, gamma_val, _, _, sign_val), items in groups.items():
        # If duplicates exist for the same key, pick the first deterministically.
        items.sort(key=lambda x: x[0])
        path, info = items[0]

        data = np.loadtxt(path, delimiter=",", skiprows=1)
        zeta = data[:, 2]
        q_num = data[:, 3]
        j_num = data[:, 4]

        ghd_sign = args.sign if args.sign is not None else sign_val
        q_ghd = np.array([hyd_charge_MICHELE(r_val, z, ghd_sign, gamma_val) for z in zeta], dtype=float)
        j_ghd = np.array([hyd_current_MICHELE(r_val, z, ghd_sign, gamma_val) for z in zeta], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zeta, q_num, marker="x", c="b",s=7, alpha=0.85, label=f"IT s={info['s']}")
        axes[0].plot(zeta, q_ghd, c="#D2C85DE2", lw=1.8, label="GHD")
        axes[1].scatter(zeta, j_num, marker="+", c='r',s=7, alpha=0.85, label=f"IT s={info['s']}")
        axes[1].plot(zeta, j_ghd, c='orange', lw=1.8, label="GHD")

        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=9)
            ax.set_xlim(-2.5, 2.5)
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")

        if info["beta"] == 0:
            title = rf"$r={r_val},\ \mathrm{{sign}}={sign_val},\ \gamma={gamma_val},\ T={info['T']},\ N={info['N']},\ s={info['s']}$"
            name_tag = f"NEEL_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}_s{info['s']}"
        elif info["beta"] is None and betaL_val is not None and betaR_val is not None:
            title = rf"$\beta_L={betaL_val},\ \beta_R={betaR_val},\ r={r_val},\ \mathrm{{sign}}={sign_val},\ \gamma={gamma_val},\ T={info['T']},\ N={info['N']},\ s={info['s']}$"
            name_tag = f"betaL{betaL_val}_betaR{betaR_val}_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}_s{info['s']}"
        else:
            title = rf"$\beta={info['beta']},\ r={r_val},\ \mathrm{{sign}}={sign_val},\ \gamma={gamma_val},\ T={info['T']},\ N={info['N']},\ s={info['s']}$"
            name_tag = f"Beta{info['beta']}_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_T{info['T']}_N{info['N']}_s{info['s']}"

        fig.suptitle(title)
        fig.tight_layout()
        outpng = os.path.join(png_dir, f"GHD_IT_vs_GHD_{name_tag}.png")
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outpng}")


if __name__ == "__main__":
    main()
