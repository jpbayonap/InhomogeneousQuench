#!/usr/bin/env python3
"""
Build a q-heatmap from GHD_IT_CSV outputs produced by main_dynamics_it.py / run_it.sh.

For r=0 and sign="+", the saved q-column corresponds to qp_symm(0, x, C_t, bc="open")
(because main_dynamics_it.py currently extracts profiles with bc="open").
"""

import argparse
import os
import re
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def parse_list(s, cast=float):
    if s is None:
        return None
    return [cast(x) for x in s.replace(",", " ").split()]


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
            "beta": None,
            "betaL": None,
            "betaR": None,
            "m": None,
            "A": None,
            "r": int(d["r"]),
            "s": int(float(d["s"])),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
        if kind == "neel":
            out["beta"] = 0.0
        elif kind == "beta":
            out["beta"] = float(d["beta"])
        elif kind == "beta_lr":
            out["betaL"] = float(d["betaL"])
            out["betaR"] = float(d["betaR"])
        elif kind in ("phsymm", "phsymm_odd"):
            out["m"] = int(d["m"])
            out["A"] = float(d["A"])
        return out
    return None


def time_edges(tvals):
    tvals = np.asarray(tvals, dtype=float)
    if tvals.ndim != 1 or tvals.size == 0:
        raise ValueError("tvals must be a non-empty 1D array")
    if tvals.size == 1:
        dt = max(1.0, abs(tvals[0]) * 0.05)
        return np.array([tvals[0] - dt, tvals[0] + dt], dtype=float)
    mids = 0.5 * (tvals[:-1] + tvals[1:])
    first = tvals[0] - (mids[0] - tvals[0])
    last = tvals[-1] + (tvals[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def latex_tick_formatter(x, _pos=None):
    if np.isclose(x, round(x)):
        label = str(int(round(x)))
    else:
        label = f"{x:.3f}".rstrip("0").rstrip(".")
    return rf"${label}$"


def compact_num_label(x):
    if np.isclose(x, round(x)):
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def paper_charge_symbol(info):
    if info["r"] == 0 and info["sign"] == "+":
        return r"q^{0,-}"
    return rf"q^{{{info['r']},{info['sign']}}}"


def paper_current_symbol(info):
    if info["r"] == 0 and info["sign"] == "+":
        return r"J^{0,-}"
    return rf"J^{{{info['r']},{info['sign']}}}"


def resolve_usetex(requested):
    if requested and shutil.which("latex") is None:
        print("WARNING: --usetex requested but 'latex' was not found. Falling back to mathtext.")
        return False
    return bool(requested)


def collect_entries(csv_dir, args):
    entries = []
    times_filter = parse_list(args.times, float)

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        info = parse_it_name(fname)
        if info is None:
            continue
        if args.init_state is not None and info["kind"] != args.init_state:
            continue
        if args.r is not None and info["r"] != args.r:
            continue
        if args.sign is not None and info["sign"] != args.sign:
            continue
        if args.gamma is not None and not np.isclose(info["gamma"], args.gamma):
            continue
        if args.size is not None and info["N"] != 2 * args.size:
            continue
        if args.s_offset is not None and info["s"] != args.s_offset:
            continue
        if args.time_min is not None and info["T"] < args.time_min:
            continue
        if args.time_max is not None and info["T"] > args.time_max:
            continue
        if times_filter is not None and not any(np.isclose(info["T"], t) for t in times_filter):
            continue
        if args.beta is not None:
            if info["beta"] is None or not np.isclose(info["beta"], args.beta):
                continue
        if args.betaL is not None:
            if info["betaL"] is None or not np.isclose(info["betaL"], args.betaL):
                continue
        if args.betaR is not None:
            if info["betaR"] is None or not np.isclose(info["betaR"], args.betaR):
                continue
        if args.phsymm_m is not None:
            if info.get("m") is None or info["m"] != args.phsymm_m:
                continue
        if args.phsymm_A is not None:
            if info.get("A") is None or not np.isclose(info["A"], args.phsymm_A):
                continue
        entries.append((os.path.join(csv_dir, fname), info))

    return entries


def load_qj_matrices(entries):
    rows = []
    seen_times = set()
    n_expected = None

    for path, info in entries:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] < 5:
            raise ValueError(f"Unexpected CSV shape in {path}: {data.shape}")
        n_sites = data.shape[0]
        if n_expected is None:
            n_expected = n_sites
        elif n_sites != n_expected:
            raise ValueError(f"Mismatched number of rows: {path} has {n_sites}, expected {n_expected}")

        t_val = float(info["T"])
        if t_val in seen_times:
            raise ValueError(f"Duplicate time T={t_val} in filtered set; refine filters.")
        seen_times.add(t_val)
        q_vals = np.real_if_close(data[:, 3]).astype(float)
        j_vals = np.real_if_close(data[:, 4]).astype(float)
        rows.append((t_val, q_vals, j_vals, info))

    if not rows:
        raise ValueError("No rows to build heatmap.")

    rows.sort(key=lambda row: row[0])
    t_axis = np.array([r[0] for r in rows], dtype=float)
    q_xt = np.vstack([r[1] for r in rows])
    j_xt = np.vstack([r[2] for r in rows])
    info0 = rows[0][3]
    return t_axis, q_xt, j_xt, info0


def make_base_name(info, nt):
    if info["kind"] == "beta_lr":
        prefix = f"GHD_NEEL_HEAT_betaL{info['betaL']}_betaR{info['betaR']}"
    elif info["kind"] == "neel":
        prefix = "GHD_NEEL_HEAT_NEEL"
    elif info["kind"] == "vac_fill":
        prefix = "GHD_NEEL_HEAT_VAC_FILL"
    elif info["kind"] == "vac_infty":
        prefix = "GHD_NEEL_HEAT_VAC_INFTY"
    elif info["kind"] == "mixed_neel":
        prefix = "GHD_NEEL_HEAT_MIXED_NEEL"
    elif info["kind"] == "phsymm":
        prefix = f"GHD_NEEL_HEAT_PHSYMM_m{info['m']}_A{info['A']:.6g}"
    elif info["kind"] == "phsymm_odd":
        prefix = f"GHD_NEEL_HEAT_PHSYMM_ODD_m{info['m']}_A{info['A']:.6g}"
    else:
        prefix = f"GHD_NEEL_HEAT_Beta{info['beta']}"
    return (
        f"{prefix}_r{info['r']}_s_{info['s']}_sign{info['sign']}"
        f"_gamma{info['gamma']:.2f}_N{info['N']}_nt{nt}"
    )


def main():
    ap = argparse.ArgumentParser(description="Build q(x,t) heatmap from GHD_IT_CSV outputs.")
    ap.add_argument("--outdir", type=str, default=".", help="Base directory containing GHD_IT_CSV")
    ap.add_argument("--csv-dir", type=str, default=None, help="Override input CSV directory (default: <outdir>/GHD_IT_CSV)")
    ap.add_argument("--r", type=int, default=0, help="Filter r (use r=0 for qp_symm(0,x,C,...) heatmap)")
    ap.add_argument("--sign", type=str, default="+", help="Filter sign (+ or -)")
    ap.add_argument("--gamma", type=float, default=None, help="Filter a single gamma")
    ap.add_argument("--size", type=int, default=None, help="Half-size L (filters N=2L)")
    ap.add_argument("--s-offset", type=int, default=None, help="Filter s offset from GHD_IT filenames")
    ap.add_argument("--beta", type=float, default=None, help="Filter homogeneous beta (use 0.0 for NEEL)")
    ap.add_argument("--betaL", type=float, default=None, help="Filter betaL")
    ap.add_argument("--betaR", type=float, default=None, help="Filter betaR")
    ap.add_argument("--phsymm-m", type=int, default=None, help="Filter PHSYMM index m")
    ap.add_argument("--phsymm-A", type=float, default=None, help="Filter PHSYMM amplitude A")
    ap.add_argument(
        "--init-state",
        type=str,
        default=None,
        choices=["neel", "beta", "beta_lr", "vac_fill", "vac_infty", "mixed_neel", "phsymm", "phsymm_odd"],
        help="Filter by initial-state family from filename.",
    )
    ap.add_argument("--times", type=str, default=None, help="Optional list of times to include")
    ap.add_argument("--time-min", type=float, default=None, help="Minimum time")
    ap.add_argument("--time-max", type=float, default=None, help="Maximum time")
    ap.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap")
    ap.add_argument("--vmin", type=float, default=None, help="Color min")
    ap.add_argument("--vmax", type=float, default=1.0, help="Color max")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX text rendering")
    ap.add_argument("--save-pdf", action="store_true", help="Also save a PDF copy of the figure")
    ap.add_argument("--show-title", action="store_true", help="Show plot title")
    ap.add_argument("--no-show-title", dest="show_title", action="store_false", help="Hide plot title")
    ap.set_defaults(show_title=False)
    args = ap.parse_args()

    use_tex = resolve_usetex(args.usetex)

    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 15,
    })

    csv_dir = args.csv_dir or os.path.join(args.outdir, "GHD_IT_CSV")
    if not os.path.isdir(csv_dir):
        raise SystemExit(f"Missing CSV directory: {csv_dir}")

    entries = collect_entries(csv_dir, args)
    if not entries:
        raise SystemExit("No matching GHD_IT CSV files found.")

    t_axis, q_xt, j_xt, info = load_qj_matrices(entries)
    nt, nx = q_xt.shape

    # Save heatmap matrix data.
    heat_csv_dir = os.path.join(args.outdir, "GHD_NEEL_HEAT_CSV")
    heat_png_dir = os.path.join(args.outdir, "GHD_NEEL_HEAT_PNG")
    heat_pdf_dir = os.path.join(args.outdir, "GHD_NEEL_HEAT_PDF")
    os.makedirs(heat_csv_dir, exist_ok=True)
    os.makedirs(heat_png_dir, exist_ok=True)
    if args.save_pdf:
        os.makedirs(heat_pdf_dir, exist_ok=True)

    base = make_base_name(info, nt)
    out_csv = os.path.join(heat_csv_dir, base + ".csv")
    out_png = os.path.join(heat_png_dir, base + ".png")
    out_pdf = os.path.join(heat_pdf_dir, base + ".pdf")

    header = ",".join(["time"] + [f"x{i}" for i in range(nx)])
    heat_table = np.column_stack([t_axis, q_xt])
    np.savetxt(out_csv, heat_table, delimiter=",", header=header, comments="")

    # Plot heatmap with final-time q and J profiles.
    x_edges = np.arange(nx + 1) - 0.5
    t_edges = time_edges(t_axis)
    q_last = q_xt[-1]
    j_last = j_xt[-1]
    t_last = t_axis[-1]
    center = info["N"] // 2
    x_vals = np.arange(nx, dtype=float)
    zeta_vals = (x_vals - (center - 1)) / t_last

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        gridspec_kw={"width_ratios": [1.8, 1.0, 1.0]},
    )
    heat_ax, q_ax, j_ax = axes

    im = heat_ax.pcolormesh(
        x_edges,
        t_edges,
        q_xt,
        shading="auto",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        rasterized=True,
    )
    heat_ax.set_xlabel(r"$x$")
    heat_ax.set_ylabel(r"$tJ$")
    heat_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    heat_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    heat_ax.xaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))
    heat_ax.yaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))

    bc_txt = r"\mathrm{open}"
    if info["kind"] == "beta_lr":
        beta_txt = rf"\beta_L={info['betaL']},\ \beta_R={info['betaR']}"
    elif info["kind"] == "neel":
        beta_txt = r"\mathrm{NEEL}"
    elif info["kind"] == "vac_fill":
        beta_txt = r"\mathrm{Vac/Fill}"
    elif info["kind"] == "vac_infty":
        beta_txt = r"\mathrm{Vac/Infty}"
    elif info["kind"] == "mixed_neel":
        beta_txt = r"\mathrm{Mixed\ Neel}"
    elif info["kind"] == "phsymm":
        beta_txt = rf"\mathrm{{PHSYMM}},\ m={info['m']},\ A={info['A']:.6g}"
    elif info["kind"] == "phsymm_odd":
        beta_txt = rf"\mathrm{{PHSYMM\ odd}},\ m={info['m']},\ A={info['A']:.6g}"
    else:
        beta_txt = rf"\beta={info['beta']}"
    title = (
        rf"${beta_txt},\ r={info['r']},\ \mathrm{{sign}}={info['sign']},\ "
        rf"\gamma={info['gamma']:.2f},\ s={info['s']},\ N={info['N']},\ "
        rf"q_p^{{(r=0)}}(x,t),\ \mathrm{{bc}}={bc_txt}$"
    )
    # Keep title compact if r != 0 or sign != + (still allowed by the filters)
    if not (info["r"] == 0 and info["sign"] == "+"):
        title = (
            rf"${beta_txt},\ r={info['r']},\ \mathrm{{sign}}={info['sign']},\ "
            rf"\gamma={info['gamma']:.2f},\ s={info['s']},\ N={info['N']}$"
        )
    if args.show_title:
        heat_ax.set_title(title)

    cbar = fig.colorbar(im, ax=heat_ax)
    charge_symbol = paper_charge_symbol(info)
    current_symbol = paper_current_symbol(info)
    cbar.set_label(rf"$\langle {charge_symbol}_x(t) \rangle$")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))

    t_last_label = compact_num_label(t_last)

    q_ax.plot(zeta_vals, q_last, color="tab:blue", lw=1.8)
    q_ax.set_xlabel(r"$\zeta$")
    q_ax.set_ylabel(rf"${charge_symbol}(\zeta)$")
    q_ax.set_xlim(-2.5, 2.5)
    q_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    q_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    q_ax.xaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))
    q_ax.yaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))
    q_ax.grid(True, alpha=0.25)

    j_ax.plot(zeta_vals, j_last, color="tab:red", lw=1.8)
    j_ax.set_xlabel(r"$\zeta$")
    j_ax.set_ylabel(rf"${current_symbol}(\zeta)$")
    j_ax.set_xlim(-2.5, 2.5)
    j_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    j_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    j_ax.xaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))
    j_ax.yaxis.set_major_formatter(FuncFormatter(latex_tick_formatter))
    j_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Found {nt} time slices (N={nx})")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")
    if args.save_pdf:
        print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
