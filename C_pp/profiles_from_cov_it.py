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


def main():
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
    ap.set_defaults(save_profile_png=True)
    args = ap.parse_args()

    args.sizes = parse_list(args.sizes, int)
    args.s_offsets = parse_list(args.s_offsets, int)
    args.gammas = parse_list(args.gammas, float)
    args.times = parse_list(args.times, float)
    r_list = parse_list(args.r_list, int)
    sign_list = parse_list(args.sign, str)
    invalid = [s for s in sign_list if s not in {"+", "-"}]
    if invalid:
        raise SystemExit(f"Invalid --sign values: {invalid}. Allowed values are '+' and '-'.")
    if args.z_min >= args.z_max:
        raise SystemExit("--z-min must be smaller than --z-max.")

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

        for r in r_list:
            for sign in sign_list:
                outcsv = csv_outpath(args.outdir, meta, r, sign)
                outpng = png_outpath(args.outdir, outcsv)

                q_vals, j_vals = compute_profiles(C_t, xs, r, sign)
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

                if args.save_profile_png:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
                    axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
                    axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
                    for ax in axes:
                        ax.set_xlabel(r"$\zeta$")
                        ax.grid(True, ls="--", alpha=0.5)
                        ax.legend()
                        ax.set_xlim(args.z_min, args.z_max)
                    axes[0].set_ylabel("q")
                    axes[1].set_ylabel("J")
                    for ax in axes:
                        ax.set_xlim(-2.5,2.5)
                    fig.suptitle(title_for(meta, r, sign, a_sites, b_sites))
                    fig.tight_layout()
                    fig.savefig(outpng, dpi=200)
                    plt.close(fig)
                    print(f"wrote {outcsv} and {outpng}")
                else:
                    print(f"wrote {outcsv}")
        return None

    Parallel(n_jobs=args.n_jobs)(delayed(run_entry)(meta) for meta in entries)


if __name__ == "__main__":
    main()
