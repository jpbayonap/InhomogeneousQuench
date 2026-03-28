#!/usr/bin/env python3
"""
Rebuild q/J profiles from saved covariance matrices produced by
main_dynamics_it_cov.py.

running example:
python3 profiles_from_cov_it.py \
  --outdir ./runs/neel_cov_local_L1500 \
  --cov-dir ./runs/neel_cov_local_L1500/GHD_IT_COV \
  --init-state neel \
  --sizes "1500" \
  --s-offsets "1500" \
  --gammas "1.0" \
  --times "200" \
  --r-list "1 3 5" \
  --sign "-" \
  --n-jobs 1


"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from joblib import Parallel, delayed

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)


INIT_STATES = ("neel", "neel_even", "beta", "beta_lr", "vac_fill", "mixed_neel", "vac_infty", "phsymm", "phsymm_odd")
J_HOP = 1.0


def parse_list(s, cast=float):
    if s is None:
        return None
    parts = str(s).replace(",", " ").split()
    return [cast(p) for p in parts]


def build_cov_search_dir(outdir, cov_dir):
    if cov_dir is not None:
        return cov_dir
    return os.path.join(outdir, "GHD_IT_COV")


def load_meta(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "path": npz_path,
            "gamma": float(data["gamma"]),
            "time": float(data["time"]),
            "N": int(data["N"]),
            "L": int(data["L"]),
            "center": int(data["center"]),
            "s_offset": int(data["s_offset"]),
            "init_state": str(data["init_state"].item()),
            "beta": float(data["beta"]),
            "betaL": float(data["betaL"]),
            "betaR": float(data["betaR"]),
            "phsymm_m": int(data["phsymm_m"]),
            "phsymm_A": float(data["phsymm_A"]),
        }


def matches(meta, args):
    if args.init_state is not None and meta["init_state"] != args.init_state:
        return False
    if args.sizes is not None and meta["L"] not in args.sizes:
        return False
    if args.s_offsets is not None and meta["s_offset"] not in args.s_offsets:
        return False
    if args.gammas is not None and not any(np.isclose(meta["gamma"], g) for g in args.gammas):
        return False
    if args.times is not None and not any(np.isclose(meta["time"], t) for t in args.times):
        return False
    if args.beta is not None:
        if np.isnan(meta["beta"]) or not np.isclose(meta["beta"], args.beta):
            return False
    if args.betaL is not None:
        if np.isnan(meta["betaL"]) or not np.isclose(meta["betaL"], args.betaL):
            return False
    if args.betaR is not None:
        if np.isnan(meta["betaR"]) or not np.isclose(meta["betaR"], args.betaR):
            return False
    if args.phsymm_m is not None and meta["phsymm_m"] != args.phsymm_m:
        return False
    if args.phsymm_A is not None and not np.isclose(meta["phsymm_A"], args.phsymm_A):
        return False
    return True


def csv_outpath(base_outdir, meta, r, sign):
    csv_dir = os.path.join(base_outdir, "GHD_IT_CSV")
    os.makedirs(csv_dir, exist_ok=True)
    g = meta["gamma"]
    T = meta["time"]
    N = meta["N"]
    s_off = meta["s_offset"]
    state = meta["init_state"]
    if state == "beta_lr":
        return os.path.join(
            csv_dir,
            f"GHD_IT_betaL_{meta['betaL']}_betaR_{meta['betaR']}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "vac_fill":
        return os.path.join(
            csv_dir,
            f"GHD_vac_fill_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "neel_even":
        return os.path.join(
            csv_dir,
            f"GHD_IT_NEEL_EVEN_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "mixed_neel":
        return os.path.join(
            csv_dir,
            f"GHD_IT_MixedNeel_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "vac_infty":
        return os.path.join(
            csv_dir,
            f"GHD_IT_VacInfty_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "phsymm_odd":
        return os.path.join(
            csv_dir,
            f"GHD_IT_PHSYMM_ODD_m{meta['phsymm_m']}_A{meta['phsymm_A']:.6g}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "phsymm":
        return os.path.join(
            csv_dir,
            f"GHD_IT_PHSYMM_m{meta['phsymm_m']}_A{meta['phsymm_A']:.6g}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    if state == "neel":
        return os.path.join(
            csv_dir,
            f"GHD_IT_NEEL_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
    return os.path.join(
        csv_dir,
        f"GHD_IT_Beta_{meta['beta']}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
    )


def png_outpath(base_outdir, csv_path):
    png_dir = os.path.join(base_outdir, "GHD_IT_PNG")
    os.makedirs(png_dir, exist_ok=True)
    return os.path.join(png_dir, os.path.basename(csv_path).replace(".csv", ".png"))


def xgrid_outpaths(base_outdir, meta, r_list, sign_label, x_pad):
    state = meta["init_state"]
    png_dir = os.path.join(base_outdir, "GHD_IT_X_PNG", state)
    pdf_dir = os.path.join(base_outdir, "GHD_IT_X_PDF", state)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    r_tag = "r" + "_".join(str(int(r)) for r in r_list)
    s_tag = f"s_{meta['s_offset']}"
    base = f"GHD_IT_{state.upper()}_xgrid_{sign_label}_{r_tag}_{s_tag}_gamma{meta['gamma']:.2f}_T{meta['time']}_N{meta['N']}_xpad{x_pad}"
    return (
        os.path.join(png_dir, base + ".png"),
        os.path.join(pdf_dir, base + ".pdf"),
    )


def title_for(meta, r, sign, a_sites, b_sites):
    g = meta["gamma"]
    T = meta["time"]
    N = meta["N"]
    state = meta["init_state"]
    a_label = f"[{int(a_sites[0])}, {int(a_sites[-1])}]"
    b_label = f"[{int(b_sites[0])}, {int(b_sites[-1])}]"
    if state == "beta_lr":
        return rf"$\beta_L={meta['betaL']},\ \beta_R={meta['betaR']},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "vac_fill":
        return rf"$\mathrm{{Vac/Fill}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "neel_even":
        return rf"$\mathrm{{Vacuum\ Neel\ even}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "mixed_neel":
        return rf"$\mathrm{{Mixed\ Neel}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "vac_infty":
        return rf"$\mathrm{{Vac/Infty}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "phsymm_odd":
        return rf"$\mathrm{{PHSYMM\ odd}},\ m={meta['phsymm_m']},\ A={meta['phsymm_A']:.6g},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "phsymm":
        return rf"$\mathrm{{PHSYMM}},\ m={meta['phsymm_m']},\ A={meta['phsymm_A']:.6g},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    if state == "neel":
        return rf"$A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"
    return rf"$\beta={meta['beta']},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$"


def xgrid_title_for(meta, r_list, sign_label):
    g = meta["gamma"]
    T = meta["time"]
    N = meta["N"]
    state = meta["init_state"]
    r_tag = ",".join(str(int(r)) for r in r_list)
    if state == "neel":
        return rf"$\mathrm{{Neel}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "neel_even":
        return rf"$\mathrm{{Vacuum\ Neel\ even}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "vac_fill":
        return rf"$\mathrm{{Vac/Fill}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "mixed_neel":
        return rf"$\mathrm{{Mixed\ Neel}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "vac_infty":
        return rf"$\mathrm{{Vac/Infty}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "phsymm":
        return rf"$\mathrm{{PHSYMM}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "phsymm_odd":
        return rf"$\mathrm{{PHSYMM\ odd}},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    if state == "beta_lr":
        return rf"$\beta_L={meta['betaL']},\ \beta_R={meta['betaR']},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"
    return rf"$\beta={meta['beta']},\ {sign_label},\ r\in\{{{r_tag}\}},\ \gamma={g},\ T={T},\ N={N}$"


def get_elem(C, i, j, bc):
    N = C.shape[0]
    ii = int(i)
    jj = int(j)
    if bc and bc[0].lower() == "p":
        return C[ii % N, jj % N]
    if 0 <= ii < N and 0 <= jj < N:
        return C[ii, jj]
    return 0.0 + 0.0j


def qp_left(r, x, C, bc="pbc", J=J_HOP):
    return J * (get_elem(C, x, x + r, bc) + get_elem(C, x + r, x, bc))


def qm_left(r, x, C, bc="pbc", J=J_HOP):
    return 1j * J * (get_elem(C, x, x + r, bc) - get_elem(C, x + r, x, bc))


def jp_left(r, x, C, bc="pbc", J=J_HOP):
    J_sq = J**2
    return 1j * J_sq * (
        get_elem(C, x + 1, x + r, bc)
        - get_elem(C, x, x + r + 1, bc)
        + get_elem(C, x + r + 1, x, bc)
        - get_elem(C, x + r, x + 1, bc)
    )


def jm_left(r, x, C, bc="pbc", J=J_HOP):
    J_sq = J**2
    return -J_sq * (
        get_elem(C, x + 1, x + r, bc)
        - get_elem(C, x, x + r + 1, bc)
        - get_elem(C, x + r + 1, x, bc)
        + get_elem(C, x + r, x + 1, bc)
    )


def compute_profiles(C_t, xs, r, sign):
    if sign == "-":
        q_vals = np.array([np.real(qm_left(r, x, C_t, "open")) for x in xs])
        j_vals = np.array([np.real(jm_left(r, x, C_t, "open")) for x in xs])
    else:
        q_vals = np.array([np.real(qp_left(r, x, C_t, "open")) for x in xs])
        j_vals = np.array([np.real(jp_left(r, x, C_t, "open")) for x in xs])
    return q_vals, j_vals


def qx_ylabel(sign):
    if sign == "-":
        return r"$\langle q^{(r,-)}_x(t)\rangle$"
    return r"$\langle q^{(r,+)}_x(t)\rangle$"


def build_xgrid_rows(r_list, sign_list, parity_mode):
    if parity_mode:
        return [
            ("+", [r for r in r_list if int(r) % 2 == 0]),
            ("-", [r for r in r_list if int(r) % 2 != 0]),
        ]
    return [(sign, list(r_list)) for sign in sign_list]


def save_xgrid(base_outdir, meta, xs, center, q_lookup, r_list, sign_list, x_pad, parity_mode):
    row_specs = build_xgrid_rows(r_list, sign_list, parity_mode)
    row_specs = [(sign, rs) for sign, rs in row_specs if rs]
    if not row_specs:
        return

    cols = max(len(rs) for _, rs in row_specs)
    rows = len(row_specs)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.4 * rows), squeeze=False, sharex=False)

    x_rel = np.asarray(xs, dtype=int) - (int(center) - 1)
    half_width = int(x_pad)
    mask = (x_rel >= -half_width) & (x_rel <= half_width)

    for row_idx, (sign, rs) in enumerate(row_specs):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(rs):
                ax.axis("off")
                continue
            r = rs[col_idx]
            q_vals = q_lookup.get((r, sign))
            if q_vals is None:
                ax.axis("off")
                continue
            q_window = np.asarray(q_vals[mask], dtype=float)
            ax.scatter(x_rel[mask], q_window, s=12, c="tab:blue", alpha=0.75, linewidths=0)
            ax.set_xlim(-half_width - 0.5, half_width + 0.5)
            if int(r) == 1:
                y_center = round(float(np.mean(q_window)), 4)
                y_halfspan = 1.5e-4
                ax.set_ylim(y_center - y_halfspan, y_center + y_halfspan)
            ax.set_title(rf"$r={r},\ \mathrm{{sign}}={sign}$", fontsize=12, pad=6)
            ax.grid(True, ls="--", alpha=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            if row_idx == rows - 1:
                ax.set_xlabel(r"$x$")
            if col_idx == 0:
                ax.set_ylabel(qx_ylabel(sign))

    sign_label = r"\mathrm{parity\ paired}" if parity_mode else r"\mathrm{all\ selected\ signs}"
    fig.suptitle(xgrid_title_for(meta, r_list, sign_label), fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sign_file_tag = "pmparity" if parity_mode else "signs_" + "_".join(s.replace("+", "p").replace("-", "m") for s in sign_list)
    outpng, outpdf = xgrid_outpaths(base_outdir, meta, r_list, sign_file_tag, x_pad)
    fig.savefig(outpng, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outpdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {outpng}")
    print(f"wrote {outpdf}")


def main():
    # Match the publication-style LaTeX rendering used by the superposed plotting scripts.
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
    })

    ap = argparse.ArgumentParser(description="Rebuild q/J profiles from saved covariance matrices.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory")
    ap.add_argument("--cov-dir", type=str, default=None, help="Directory with covariance .npz files (default: <outdir>/GHD_IT_COV)")
    ap.add_argument("--init-state", type=str, default=None, choices=INIT_STATES, help="Filter by initial-state family.")
    ap.add_argument("--sizes", type=str, default=None, help="Comma/space-separated half-sizes (L).")
    ap.add_argument("--s-offsets", type=str, default=None, help="Comma/space-separated s values.")
    ap.add_argument("--gammas", type=str, default=None, help="Comma/space-separated gamma values.")
    ap.add_argument("--times", type=str, default=None, help="Comma/space-separated times.")
    ap.add_argument("--beta", type=float, default=None, help="Filter by homogeneous beta.")
    ap.add_argument("--betaL", type=float, default=None, help="Filter by betaL.")
    ap.add_argument("--betaR", type=float, default=None, help="Filter by betaR.")
    ap.add_argument("--phsymm-m", type=int, default=None, help="Filter by PHSYMM index m.")
    ap.add_argument("--phsymm-A", type=float, default=None, help="Filter by PHSYMM amplitude A.")
    ap.add_argument("--r-list", type=str, required=True, help="Comma/space-separated r values.")
    ap.add_argument("--sign", type=str, required=True, help="One or more signs, e.g. '-' or '+ -'.")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over covariance files.")
    ap.add_argument("--save-profile-png", dest="save_profile_png", action="store_true", help="Save per-profile PNGs (default: on).")
    ap.add_argument("--no-save-profile-png", dest="save_profile_png", action="store_false", help="Disable PNG generation.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    ap.add_argument("--z-min", type=float, default=-4.0, help="Minimum zeta shown in profile PNGs.")
    ap.add_argument("--z-max", type=float, default=4.0, help="Maximum zeta shown in profile PNGs.")
    ap.add_argument("--x-pad", type=int, default=None, help="If set, build x-space subplot grids using a fixed window |x| <= x_pad instead of standard zeta PNGs.")
    ap.set_defaults(save_profile_png=True)
    args = ap.parse_args()

    args.sizes = parse_list(args.sizes, int)
    args.s_offsets = parse_list(args.s_offsets, int)
    args.gammas = parse_list(args.gammas, float)
    args.times = parse_list(args.times, float)
    r_list = parse_list(args.r_list, int)
    parity_sign_mode = str(args.sign).strip() == "+/-"
    if parity_sign_mode:
        sign_list = ["+", "-"]
    else:
        sign_list = parse_list(args.sign, str)
    invalid = [s for s in sign_list if s not in {"+", "-"}]
    if invalid:
        raise SystemExit(f"Invalid --sign values: {invalid}. Allowed values are '+' and '-'.")
    if args.z_min >= args.z_max:
        raise SystemExit("--z-min must be smaller than --z-max.")
    if args.x_pad is not None and args.x_pad < 0:
        raise SystemExit("--x-pad must be non-negative.")

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

    def run_entry(meta):
        with np.load(meta["path"], allow_pickle=False) as data:
            C_t = data["C_t"]
            xs = np.asarray(data["xs"], dtype=int)
            # Recompute zeta from x using the left-interface convention:
            # zeta = (x - (center-1)) / T, so the last left site is zeta=0.
            zetas = (np.asarray(data["xs"], dtype=int) - (int(data["center"]) - 1)) / float(meta["time"])
            a_sites = np.asarray(data["a_sites"], dtype=int)
            b_sites = np.asarray(data["b_sites"], dtype=int)
            center = int(data["center"])

        q_lookup = {}

        for r in r_list:
            for sign in sign_list:
                outcsv = csv_outpath(args.outdir, meta, r, sign)
                outpng = png_outpath(args.outdir, outcsv)

                q_vals, j_vals = compute_profiles(C_t, xs, r, sign)
                q_lookup[(r, sign)] = q_vals
                if args.skip_existing and os.path.isfile(outcsv) and (not args.save_profile_png or os.path.isfile(outpng)):
                    print(f"skip existing {outcsv}")
                    continue
                np.savetxt(
                    outcsv,
                    np.column_stack([
                        np.full_like(zetas, meta["gamma"], dtype=float),
                        np.full_like(zetas, meta["time"], dtype=float),
                        zetas,
                        q_vals,
                        j_vals,
                        xs,
                    ]),
                    delimiter=",",
                    header="gamma,time,zeta,q,j,x",
                    comments="",
                )

                if args.save_profile_png and args.x_pad is None:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
                    axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
                    axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
                    for ax in axes:
                        ax.set_xlabel(r"$\zeta$")
                        ax.grid(True, ls="--", alpha=0.5)
                        ax.legend()
                        ax.set_xlim(args.z_min, args.z_max)
                    axes[0].set_ylabel(r"$q(\zeta)$")
                    axes[1].set_ylabel(r"$J(\zeta)$")
                    for ax in axes:
                        ax.set_xlim(-2.5,2.5)
                    fig.suptitle(title_for(meta, r, sign, a_sites, b_sites))
                    fig.tight_layout()
                    fig.savefig(outpng, dpi=200)
                    plt.close(fig)
                    print(f"wrote {outcsv} and {outpng}")
                else:
                    print(f"wrote {outcsv}")

        if args.save_profile_png and args.x_pad is not None:
            save_xgrid(args.outdir, meta, xs, center, q_lookup, r_list, sign_list, args.x_pad, parity_sign_mode)
        return None

    Parallel(n_jobs=args.n_jobs)(delayed(run_entry)(meta) for meta in entries)


if __name__ == "__main__":
    main()
