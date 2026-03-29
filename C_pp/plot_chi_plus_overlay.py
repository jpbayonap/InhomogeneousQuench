#!/usr/bin/env python3
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_list(s, cast=str):
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    return [cast(p) for p in parts]


def parse_delta_tag_value(tag):
    m = re.fullmatch(r"delta(\d+)", str(tag))
    return int(m.group(1)) if m else math.inf


def format_delta_tag_tex(tag):
    m = re.fullmatch(r"delta(\d+)", str(tag))
    if m:
        return rf"\delta_{{{m.group(1)}}}"
    return rf"\mathrm{{{tag}}}"


def fmt_float_tag(x):
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def build_default_chi_dir(outdir, state):
    state_map = {
        "neel": os.path.join(outdir, "GHD_NEEL_HYBRID", "chi_plus", "csv"),
        "polarized": os.path.join(outdir, "GHD_POLARIZED_HYBRID", "chi_plus", "csv"),
    }
    return state_map[state]


def parse_name(path):
    name = os.path.basename(path)
    pattern = re.compile(
        r"CHI_PLUS_(?P<state>NEEL|POLARIZED)_(?P<delta_tag>delta\d+)_M(?P<M>\d+)_gamma(?P<gamma>[-0-9.]+)_wp(?P<wp>[-0-9.]+)\.csv$"
    )
    m = pattern.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "state": d["state"].lower(),
        "delta_tag": d["delta_tag"],
        "M": int(d["M"]),
        "gamma": float(d["gamma"]),
        "wp": float(d["wp"]),
    }


def load_chi_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def main():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
        }
    )

    ap = argparse.ArgumentParser(
        description="Overlay ChiPlus(k) curves for multiple delta tags and gamma values."
    )
    ap.add_argument("--outdir", type=str, default=".", help="Repo/output base directory.")
    ap.add_argument(
        "--chi-csv-dir",
        type=str,
        default=None,
        help="Directory with CHI_PLUS CSVs. Default: <outdir>/GHD_<STATE>_HYBRID/chi_plus/csv",
    )
    ap.add_argument(
        "--state",
        type=str,
        required=True,
        choices=["neel", "polarized"],
        help="Hybrid state family.",
    )
    ap.add_argument(
        "--delta-tags",
        type=str,
        required=True,
        help="Comma/space-separated delta tags, e.g. 'delta1 delta3'.",
    )
    ap.add_argument(
        "--gammas",
        type=str,
        required=True,
        help="Comma/space-separated gamma values, e.g. '0.0 1.0'.",
    )
    ap.add_argument("--M", type=int, default=None, help="Optional M filter.")
    ap.add_argument("--wp", type=float, default=None, help="Optional wp filter.")
    ap.add_argument("--k-min", type=float, default=0.0, help="Minimum k shown. Default: 0.")
    ap.add_argument("--k-max", type=float, default=float(np.pi), help="Maximum k shown. Default: pi.")
    ap.add_argument("--show-title", dest="show_title", action="store_true", help="Show figure title.")
    ap.add_argument("--no-show-title", dest="show_title", action="store_false", help="Hide figure title.")
    ap.set_defaults(show_title=True)
    args = ap.parse_args()

    if args.k_min >= args.k_max:
        raise SystemExit("--k-min must be smaller than --k-max.")

    delta_tags = sorted(set(parse_list(args.delta_tags, str)), key=parse_delta_tag_value)
    gammas = sorted(set(parse_list(args.gammas, float)))
    chi_csv_dir = args.chi_csv_dir or build_default_chi_dir(args.outdir, args.state)
    if not os.path.isdir(chi_csv_dir):
        raise SystemExit(f"Missing chi-plus directory: {chi_csv_dir}")

    entries = []
    for fname in os.listdir(chi_csv_dir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(chi_csv_dir, fname)
        info = parse_name(path)
        if info is None or info["state"] != args.state:
            continue
        if info["delta_tag"] not in delta_tags:
            continue
        if not any(np.isclose(info["gamma"], g) for g in gammas):
            continue
        if args.M is not None and info["M"] != args.M:
            continue
        if args.wp is not None and not np.isclose(info["wp"], args.wp):
            continue
        entries.append((path, info))

    if not entries:
        raise SystemExit(
            "No matching CHI_PLUS CSVs found for the requested state/delta/gamma filters."
        )

    chosen = {}
    for path, info in sorted(entries, key=lambda item: (item[1]["M"], item[1]["wp"])):
        chosen[(info["delta_tag"], info["gamma"])] = (path, info)

    missing = [
        (delta_tag, gamma)
        for delta_tag in delta_tags
        for gamma in gammas
        if (delta_tag, gamma) not in chosen
    ]
    if missing:
        preview = ", ".join([f"({d}, gamma={g:g})" for d, g in missing[:8]])
        raise SystemExit(f"Missing CHI_PLUS curves for: {preview}")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    linestyle_cycle = ["-", "--", "-.", ":"]

    for delta_idx, delta_tag in enumerate(delta_tags):
        color = color_cycle[delta_idx % len(color_cycle)]
        for gamma_idx, gamma in enumerate(gammas):
            path, _info = chosen[(delta_tag, gamma)]
            k_vals, chi_vals = load_chi_csv(path)
            mask = (k_vals >= args.k_min) & (k_vals <= args.k_max)
            label = (
                "$"
                + r",\ ".join(
                    [format_delta_tag_tex(delta_tag), rf"\gamma={fmt_float_tag(gamma)}"]
                )
                + "$"
            )
            ax.plot(
                k_vals[mask],
                chi_vals[mask],
                color=color,
                linestyle=linestyle_cycle[gamma_idx % len(linestyle_cycle)],
                lw=1.4,
                label=label,
            )

    ax.set_xlim(args.k_min, args.k_max)
    ax.grid(True, ls="--", alpha=0.45)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\chi_+(k)$")
    ax.legend(loc="best")
    if args.show_title:
        delta_txt = ", ".join(delta_tags)
        gamma_txt = ", ".join(fmt_float_tag(g) for g in gammas)
        ax.set_title(
            rf"${args.state.title()}\ \chi_+,\ \delta\in\{{{delta_txt}\}},\ \gamma\in\{{{gamma_txt}\}}$"
        )

    png_dir = os.path.join(args.outdir, "GHD_SUPERPOSED", "png", args.state)
    pdf_dir = os.path.join(args.outdir, "GHD_SUPERPOSED", "pdf", args.state)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    gamma_tag = f"g{len(gammas)}" if len(gammas) > 1 else f"gamma{fmt_float_tag(gammas[0])}"
    delta_tag = "_".join(delta_tags)
    if np.isclose(args.k_min, 0.0) and np.isclose(args.k_max, np.pi):
        k_tag = "kpos"
    else:
        k_tag = f"k{fmt_float_tag(args.k_min)}_to_{fmt_float_tag(args.k_max)}"
    file_stem = f"{args.state}_chi_plus_overlay_{delta_tag}_{gamma_tag}_{k_tag}"

    out_png = os.path.join(png_dir, file_stem + ".png")
    out_pdf = os.path.join(pdf_dir, file_stem + ".pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
