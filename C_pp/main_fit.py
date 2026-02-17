#!/usr/bin/env python3
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
''' 
python3 main_fit.py \
  --outdir /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp \
  --r-list "1" \
  --sign "+" \
  --gammas "1.0" \
  --sizes "1000" \
  --beta "1.0" \
  --s-offsets "1000" \
  --a-zeta 0.35 \
  --zeta-side right \
  --x-scale log \
  --allow-missing-times\
  --times "200 400 600 800 900"
'''

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


def find_time_key(available_times, target):
    for t in available_times:
        if np.isclose(t, target):
            return t
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract z_a(t) from q(zeta,t) where q(z_a,t)=a_zeta.")
    ap.add_argument("--outdir", type=str, default=".", help="Base directory containing GHD_IT_CSV.")
    ap.add_argument("--r-list", type=str, default=None, help="Space/comma-separated r values.")
    ap.add_argument("--sign", type=str, default=None, help="Filter sign (+ or -).")
    ap.add_argument("--gammas", type=str, default=None, help="Space/comma-separated gamma values.")
    ap.add_argument("--beta", type=float, default=None, help="Filter homogeneous beta.")
    ap.add_argument("--betaL", type=float, default=None, help="Filter betaL.")
    ap.add_argument("--betaR", type=float, default=None, help="Filter betaR.")
    ap.add_argument("--sizes", type=str, default=None, help="Space/comma-separated half-sizes L.")
    ap.add_argument("--times", type=str, default=None, help="Space/comma-separated times. If omitted, use all available times.")
    ap.add_argument("--s-offsets", type=str, default=None, help="Space/comma-separated s offsets.")
    ap.add_argument("--a-zeta", type=float, default=0.3, help="Target value a_zeta used in q(z_a)=a_zeta.")
    ap.add_argument("--x-scale", type=str, choices=["linear", "log", "symlog"], default="linear", help="Scale for time axis.")
    ap.add_argument(
        "--zeta-side",
        type=str,
        choices=["left", "right", "both"],
        default="left",
        help="Choose which zeta branch to search when solving q(z_a)=a_zeta.",
    )
    ap.add_argument("--allow-missing-times", action="store_true", help="Keep running even if some requested times are missing.")
    args = ap.parse_args()

    csv_dir = os.path.join(args.outdir, "GHD_IT_CSV")
    if not os.path.isdir(csv_dir):
        raise SystemExit(f"Missing CSV directory: {csv_dir}")

    r_list = parse_list(args.r_list, int)
    gamma_list = parse_list(args.gammas, float)
    size_list = parse_list(args.sizes, int)
    s_offset_list = parse_list(args.s_offsets, int)
    requested_times = parse_list(args.times, float)
    n_list = [2 * s for s in size_list] if size_list else None

    entries = []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_name(fname)
        if info is None:
            continue
        if args.beta is not None and (info["beta"] is None or not np.isclose(info["beta"], args.beta)):
            continue
        if args.betaL is not None and (info["betaL"] is None or not np.isclose(info["betaL"], args.betaL)):
            continue
        if args.betaR is not None and (info["betaR"] is None or not np.isclose(info["betaR"], args.betaR)):
            continue
        if r_list is not None and info["r"] not in r_list:
            continue
        if args.sign is not None and info["sign"] != args.sign:
            continue
        if gamma_list is not None and not any(np.isclose(info["gamma"], g) for g in gamma_list):
            continue
        if n_list is not None and info["N"] not in n_list:
            continue
        if s_offset_list is not None and info["s"] not in s_offset_list:
            continue
        entries.append((os.path.join(csv_dir, fname), info))

    if not entries:
        raise SystemExit("No matching CSV files found.")

    # Group by everything except time; each group should produce one z_a(t) curve.
    groups = {}
    for path, info in entries:
        key = (
            info["beta"],
            info["betaL"],
            info["betaR"],
            info["r"],
            info["sign"],
            info["gamma"],
            info["s"],
            info["N"],
        )
        groups.setdefault(key, []).append((path, info))

    png_dir = os.path.join(args.outdir, "FIT_GHD_IT_PNG")
    csv_out_dir = os.path.join(args.outdir, "FIT_GHD_IT_CSV")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(csv_out_dir, exist_ok=True)

    for key, items in groups.items():
        # Map available time -> file
        by_time = {}
        for path, info in items:
            by_time[info["T"]] = (path, info)
        available_times = sorted(by_time.keys())

        if requested_times is None:
            t_list = available_times
        else:
            t_list = requested_times

        z_a_vals = []
        q_at_za_vals = []
        t_ok = []
        missing = []

        for t_req in t_list:
            t_key = find_time_key(available_times, t_req)
            if t_key is None:
                missing.append(t_req)
                continue

            path, info = by_time[t_key]
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data[None, :]
            zeta = data[:, 2]
            q_vals = data[:, 3]

            # side mask
            if args.zeta_side == "left":
                side = zeta <= 0.0

            elif args.zeta_side == "right":
                side = zeta >= 0.0
            
            else:
                side = np.ones_like(zeta,dtype=bool)

            # guard against empty side
            if not np.any(side):
                missing.append(t_req)
                continue

            zeta_side= zeta[side]
            q_side = q_vals[side]

            idx_local = int(np.argmin(np.abs(q_side - args.a_zeta)))
            t_ok.append(float(t_key))
            z_a_val = float(zeta_side[idx_local])
            z_a_vals.append(z_a_val)
            q_at_za = float(q_side[idx_local])
            q_at_za_vals.append(q_at_za)

        if missing and not args.allow_missing_times:
            raise SystemExit(
                "Missing requested times for one group. "
                f"Group={key}, missing={missing}. "
                "Use --allow-missing-times to skip them."
            )

        if not t_ok:
            continue

        t_ok = np.array(t_ok, dtype=float)
        z_a_vals = np.array(z_a_vals, dtype=float)
        q_at_za_vals = np.array(q_at_za_vals, dtype=float)

        # Log-fit (optional): z_a(t) = A + B log(t), only for t>0.
        fit_ok = False
        mask = t_ok > 0.0
        if np.count_nonzero(mask) >= 2:
            t_fit = t_ok[mask]
            z_fit = z_a_vals[mask]
            order = np.argsort(t_fit)
            t_fit = t_fit[order]
            z_fit = z_fit[order]

            X = np.column_stack([np.ones_like(t_fit), np.log(t_fit)])
            (A, B), *_ = np.linalg.lstsq(X, z_fit, rcond=None)
            t_dense = np.linspace(t_fit.min(), t_fit.max(), 300)
            z_dense = A + B * np.log(t_dense)
            fit_ok = True

        beta, betaL, betaR, r_val, sign_val, gamma_val, s_val, N_val = key
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t_ok, z_a_vals, lw=1.6, alpha=0.6, label=r"$z_a(t)$")
        ax.scatter(t_ok, z_a_vals, s=20)
        if fit_ok:
            ax.plot(t_dense, z_dense, "k--", lw=1.8, label=rf"fit: $A+B\log t$")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$z_a$")
        if args.x_scale == "log":
            if np.any(t_ok <= 0.0):
                print("[plot] requested --x-scale log but t has non-positive values; using symlog instead")
                ax.set_xscale("symlog", linthresh=1e-3)
            else:
                ax.set_xscale("log")
        elif args.x_scale == "symlog":
            ax.set_xscale("symlog", linthresh=1e-3)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        if fit_ok:
            z_pred = A + B * np.log(t_fit)
            rmse = np.sqrt(np.mean((z_fit - z_pred) ** 2))
            print(f"[fit] A={A:.6g}, B={B:.6g}, RMSE={rmse:.3e}")
        else:
            print("[fit] skipped: need at least two positive-time points")



        if beta is None:
            title = rf"$\beta_L={betaL},\ \beta_R={betaR},\ r={r_val},\ {sign_val},\ \gamma={gamma_val},\ s={s_val},\ N={N_val},\ a_\zeta={args.a_zeta}$"
            tag = f"betaL{betaL}_betaR{betaR}_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_s{s_val}_N{N_val}"
        elif np.isclose(beta, 0.0):
            title = rf"$\mathrm{{NEEL}},\ r={r_val},\ {sign_val},\ \gamma={gamma_val},\ s={s_val},\ N={N_val},\ a_\zeta={args.a_zeta}$"
            tag = f"NEEL_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_s{s_val}_N{N_val}"
        else:
            title = rf"$\beta={beta},\ r={r_val},\ {sign_val},\ \gamma={gamma_val},\ s={s_val},\ N={N_val},\ a_\zeta={args.a_zeta}$"
            tag = f"Beta{beta}_r{r_val}_sign{sign_val}_gamma{gamma_val:.2f}_s{s_val}_N{N_val}"

        fig.suptitle(title)
        fig.tight_layout()

        outpng = os.path.join(png_dir, f"FIT_GHD_IT_za_vs_t_{tag}.png")
        outcsv = os.path.join(csv_out_dir, f"FIT_GHD_IT_za_vs_t_{tag}.csv")
        fig.savefig(outpng, dpi=200)
        plt.close(fig)

        np.savetxt(
            outcsv,
            np.column_stack([t_ok, z_a_vals, q_at_za_vals]),
            delimiter=",",
            header="t,z_a,q_at_z_a",
            comments="",
        )
        print(f"wrote {outcsv} and {outpng}")

   


if __name__ == "__main__":
    main()
