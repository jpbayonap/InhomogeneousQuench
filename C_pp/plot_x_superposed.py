#!/usr/bin/env python3
import argparse
import math
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

INIT_STATES = ("neel", "polarized", "neel_even", "beta", "beta_lr", "vac_fill", "mixed_neel", "vac_infty", "phsymm", "phsymm_odd")

'''
run example:
python3 plot_neelGHD_superposed.py \
  --outdir /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp \
  --py-csv-dir /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/GHD_SUPERPOSED/csv/vac_fill \
  --init-state vac_fill \
  --r-list "1" \
  --sign "+" \
  --gammas "1.0" \
  --sizes "1000" \
  --beta "1.0" \
  --times "200 400 600 800" \
  --group-mode r_gamma \
  --a-offset 1000 \
  --b-offsets "1000"


'''
DEFAULT_X_PAD = 20


def parse_list(s, cast=float):
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    return [cast(p) for p in parts]


def fmt2(x):
    """Format float with at most 2 digits after decimal point."""
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def fmt_int_or_float(x):
    """Use integer formatting when possible, otherwise keep compact float."""
    xf = float(x)
    if np.isclose(xf, round(xf)):
        return str(int(round(xf)))
    return fmt2(xf)


def format_region_tag(info):
    """
    Build a filename tag for spatial offsets.
    Priority for IT files is s-offset; fallback to b/a tags for legacy files.
    """
    if info.get("s") is not None:
        return f"_s{fmt_int_or_float(info['s'])}"
    if info.get("b") is not None:
        return f"_b{fmt_int_or_float(info['b'])}"
    if info.get("a") is not None:
        return f"_a{fmt_int_or_float(info['a'])}"
    return ""


def init_state_tex_label(init_state):
    labels = {
        "neel": r"\mathrm{num}",
        "polarized": r"\mathrm{num}",
        "vac_fill": r"\mathrm{num}",
        "neel_even": r"\mathrm{num}",
        "mixed_neel": r"\mathrm{num}",
        "vac_infty": r"\mathrm{num}",
        "phsymm": r"\mathrm{num}",
        "phsymm_odd": r"\mathrm{num}",
        "beta": r"\mathrm{num}",
        "beta_lr": r"\mathrm{num}",
    }
    return labels.get(init_state, r"\mathrm{Unknown}")


def init_state_panel_label(init_state):
    labels = {
        "neel": r"\mathrm{Vacuum\ Neel}",
        "polarized": r"\mathrm{Polarized}",
        "vac_fill": r"\mathrm{Vac/Fill}",
        "neel_even": r"\mathrm{Vacuum\ Neel\ even}",
        "mixed_neel": r"\mathrm{Mixed\ Neel}",
        "vac_infty": r"\mathrm{Vac/Infty}",
        "phsymm": r"\mathrm{PHSYMM}",
        "phsymm_odd": r"\mathrm{PHSYMM\ odd}",
        "beta": r"\mathrm{Beta}",
        "beta_lr": r"\mathrm{Beta\ LR}",
    }
    return labels.get(init_state, r"\mathrm{Unknown}")


def build_mat_search_dirs(outdir, mat_csv_dir, init_state):
    if mat_csv_dir is not None:
        return [mat_csv_dir]
    candidates = []
    if init_state == "neel":
        candidates.append(os.path.join(outdir, "GHD_NEEL_MAT_CSV"))
    # Default vac/fill Mathematica export location.
    if init_state == "vac_fill":
        candidates.append(os.path.join(outdir, "GHD_vac_fill_mat", "csv"))
    # Default PHSYMM Mathematica export location.
    if init_state == "phsymm":
        candidates.append(os.path.join(outdir, "GHD_PHSYMM_mat", "csv"))
    # Default odd-PHSYMM Mathematica export location.
    if init_state == "phsymm_odd":
        candidates.append(os.path.join(outdir, "GHD_PHSYMM_ODD_mat", "csv"))
    # Default Neel Mathematica export location (legacy/current).
    candidates.append(os.path.join(outdir, "GHD_NELL_MAT_CSV"))
    return [d for d in candidates if os.path.isdir(d)]


def vac_fill_unitary_ghd(zetas, r, sign, nk=4096):
    """
    Build unitary (gamma=0) GHD profiles for Vac/Fill:
      n_zeta(k) = nL(k) Theta(eps'(k)-zeta) + nR(k) Theta(zeta-eps'(k))
      nL=0, nR=1/(2pi), eps'(k)=2 sin(k)
    """
    z = np.asarray(zetas, dtype=float)
    k = np.linspace(-np.pi, np.pi, int(nk), endpoint=False)
    epsp = 2.0 * np.sin(k)
    z_col = z[:, None]
    eps_col = epsp[None, :]
    theta = np.where(z_col > eps_col, 1.0, np.where(z_col < eps_col, 0.0, 0.5))
    n_zeta = theta / (2.0 * np.pi)

    if sign == "-":
        q_mode = -2.0 * np.sin(int(r) * k)
    else:
        q_mode = 2.0 * np.cos(int(r) * k)
    j_mode = 2.0 * np.sin(k) * q_mode

    q_vals = np.trapz(n_zeta * q_mode[None, :], x=k, axis=1)
    j_vals = np.trapz(n_zeta * j_mode[None, :], x=k, axis=1)
    return q_vals, j_vals


def ensure_vac_fill_unitary_csv(mat_dir, r, sign, zetas):
    os.makedirs(mat_dir, exist_ok=True)
    out_csv = os.path.join(
        mat_dir,
        f"GHD_vac_fill_r{int(r)}_sign{sign}_M0_gamma0.00unitary.csv",
    )
    if not os.path.isfile(out_csv):
        q_vals, j_vals = vac_fill_unitary_ghd(zetas, r, sign)
        np.savetxt(out_csv, np.column_stack([zetas, q_vals, j_vals]), delimiter=",")
        print(f"wrote {out_csv}")
    return out_csv


def build_csv_search_dirs(outdir, py_csv_dir, init_state):
    if py_csv_dir is not None:
        return [py_csv_dir]

    candidates = []
    superposed_csv_root = os.path.join(outdir, "GHD_SUPERPOSED", "csv")
    if os.path.isdir(superposed_csv_root):
        if init_state is not None:
            state_dir = os.path.join(superposed_csv_root, init_state)
            if os.path.isdir(state_dir):
                candidates.append(state_dir)
        state_dirs = [os.path.join(superposed_csv_root, s) for s in INIT_STATES if os.path.isdir(os.path.join(superposed_csv_root, s))]
        candidates.extend(state_dirs)
        candidates.append(superposed_csv_root)

    # Backward compatibility with the old layout.
    candidates.extend([
        os.path.join(outdir, "GHD_SIMP_NELL_CSV"),
        os.path.join(outdir, "GHD_IT_CSV"),
    ])
    ordered = []
    seen = set()
    for path in candidates:
        if os.path.isdir(path) and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def parse_name(path):
    name = os.path.basename(path)
    pattern_neel = re.compile(
        r"GHD_NEEL_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_neel_simple = re.compile(
        r"GHD_NEEL_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_neel = re.compile(
        r"GHD_IT_NEEL_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_neel_even = re.compile(
        r"GHD_IT_NEEL_EVEN_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_vac_fill = re.compile(
        r"GHD_vac_fill_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_mixed_neel = re.compile(
        r"GHD_IT_MixedNeel_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_vac_infty = re.compile(
        r"GHD_IT_VacInfty_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_phsymm = re.compile(
        r"GHD_IT_PHSYMM_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_phsymm_odd = re.compile(
        r"GHD_IT_PHSYMM_ODD_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_beta = re.compile(
        r"GHD_IT_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_it_beta_lr = re.compile(
        r"GHD_IT_betaL_(?P<betaL>[-0-9.]+)_betaR_(?P<betaR>[-0-9.]+)_r(?P<r>[-0-9]+)_s_(?P<s>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_neel_mat = re.compile(
        r"GHD_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_M(?P<M>[-0-9]+)_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_]*)\.csv$"
    )
    pattern_hybrid_mat = re.compile(
        r"GHD_(?P<state>NEEL|POLARIZED)_HYBRID_(?P<delta_tag>delta[0-9]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_M(?P<M>[-0-9]+)_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_.-]*)\.csv$"
    )
    pattern_vac_fill_mat = re.compile(
        r"GHD_vac_fill_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_M(?P<M>[-0-9]+)_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_]*)\.csv$"
    )
    pattern_vac_fill_mat_nom = re.compile(
        r"GHD_vac_fill_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_]*)\.csv$"
    )
    pattern_phsymm_mat = re.compile(
        r"GHD_PHSYMM_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_M(?P<M>[-0-9]+)_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_]*)\.csv$"
    )
    pattern_phsymm_odd_mat = re.compile(
        r"GHD_PHSYMM_ODD_m(?P<m>[-0-9]+)_A(?P<A>[-+0-9.eE]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_M(?P<M>[-0-9]+)_gamma(?P<gamma>[-0-9.]+)(?P<tag>[A-Za-z0-9_]*)\.csv$"
    )
    pattern_beta = re.compile(
        r"GHD_NEEL_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta_simple = re.compile(
        r"GHD_NEEL_Beta_(?P<beta>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )
    pattern_beta_lr = re.compile(
        r"GHD_NEEL_betaL_(?P<betaL>[-0-9.]+)_betaR_(?P<betaR>[-0-9.]+)_r(?P<r>[-0-9]+)_a_(?P<a>[-0-9.]+)_b_(?P<b>[-0-9.]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )

    m = pattern_neel.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "neel",
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "a": int(d["a"]),
            "b": int(d["b"]),
            "s": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_neel_simple.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "neel",
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_hybrid_mat.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": d["state"].lower(),
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "T": None,
            "N": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "M": int(d["M"]),
            "delta_tag": d["delta_tag"],
            "tag": d.get("tag", "") or "",
        }

    m = pattern_neel_mat.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": "neel",
            "beta": None, "betaL": None, "betaR": None,
            "a": None, "b": None, "s": None, "T": None, "N": None,
            "r": int(d["r"]), "sign": d["sign"],
            "gamma": float(d["gamma"]), "M": int(d["M"]),
            "delta_tag": None,
            "tag": d.get("tag", "") or "",
        }

    m = pattern_vac_fill_mat.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": "vac_fill",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "T": None,
            "N": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "M": int(d["M"]),
            "tag": d.get("tag", "") or "",
        }

    m = pattern_vac_fill_mat_nom.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": "vac_fill",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "T": None,
            "N": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "M": None,
            "tag": d.get("tag", "") or "",
        }

    m = pattern_phsymm_mat.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": "phsymm",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "m": int(d["m"]),
            "A": float(d["A"]),
            "T": None,
            "N": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "M": int(d["M"]),
            "tag": d.get("tag", "") or "",
        }
    m = pattern_phsymm_odd_mat.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "mat",
            "init_state": "phsymm_odd",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "m": int(d["m"]),
            "A": float(d["A"]),
            "T": None,
            "N": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "M": int(d["M"]),
            "tag": d.get("tag", "") or "",
        }

    m = pattern_it_neel.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "neel",
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_neel_even.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "neel_even",
            "beta": 0.0,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_vac_fill.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "vac_fill",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_mixed_neel.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "mixed_neel",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_vac_infty.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "vac_infty",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_phsymm.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "phsymm",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "m": int(d["m"]),
            "A": float(d["A"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }
    m = pattern_it_phsymm_odd.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "phsymm_odd",
            "beta": None,
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "m": int(d["m"]),
            "A": float(d["A"]),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_beta_lr.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "beta_lr",
            "beta": None,
            "betaL": float(d["betaL"]),
            "betaR": float(d["betaR"]),
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_it_beta.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "beta",
            "beta": float(d["beta"]),
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": int(float(d["s"])),
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
            "M": None,
        }

    m = pattern_beta_lr.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "beta_lr",
            "beta": None,
            "betaL": float(d["betaL"]),
            "betaR": float(d["betaR"]),
            "a": int(d["a"]),
            "b": int(d["b"]),
            "s": None,
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
            "source": "py",
            "init_state": "beta",
            "beta": float(d["beta"]),
            "betaL": None,
            "betaR": None,
            "a": int(d["a"]),
            "b": int(d["b"]),
            "s": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
    m = pattern_beta_simple.match(name)
    if m:
        d = m.groupdict()
        return {
            "source": "py",
            "init_state": "beta",
            "beta": float(d["beta"]),
            "betaL": None,
            "betaR": None,
            "a": None,
            "b": None,
            "s": None,
            "r": int(d["r"]),
            "sign": d["sign"],
            "gamma": float(d["gamma"]),
            "T": float(d["T"]),
            "N": int(d["N"]),
        }
    return None


def main():
    # Publication-style defaults (requires a working LaTeX installation for text.usetex=True).
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        # Balanced sizes for a 12x5 two-panel figure (title + legend stay readable).
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
    })

    ap = argparse.ArgumentParser(description="Overlay cross-boundary q_x profiles grouped by r, gamma and/or time.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory (defaults to GHD_SUPERPOSED/{csv,png}).")
    ap.add_argument("--py-csv-dir", type=str, default=None,
                    help="Directory with Python CSVs. Default: <outdir>/GHD_SUPERPOSED/csv[/<init-state>] with fallback to legacy dirs.")
    ap.add_argument("--r-list", type=str, default=None, help="Comma/space-separated r values.")
    ap.add_argument(
        "--sign",
        type=str,
        default=None,
        help="Filter by sign. Accepts '+' or '-', or a comma/space-separated list such as '+ -'.",
    )
    ap.add_argument("--gammas", type=str, default=None, help="Comma/space-separated gamma values.")
    ap.add_argument("--beta", type=float, default=None, help="Filter by homogeneous Beta")
    ap.add_argument("--betaL", type=float, default=None, help="Filter by betaL")
    ap.add_argument("--betaR", type=float, default=None, help="Filter by betaR")
    ap.add_argument("--phsymm-m", type=int, default=None, help="Filter by PHSYMM index m")
    ap.add_argument("--phsymm-A", type=float, default=None, help="Filter by PHSYMM amplitude A")
    ap.add_argument("--sizes", type=str, default=None, help="Comma/space-separated half-sizes (L).")
    ap.add_argument("--time", type=float, default=None, help="Filter by a single T.")
    ap.add_argument("--times", type=str, default=None, help="Comma/space-separated T values to include.")
    ap.add_argument("--a-offset", type=int, default=None, help="Filter by A_OFFSET (left length).")
    ap.add_argument("--b-offsets", type=str, default=None, help="Comma/space-separated B_OFFSET values.")
    ap.add_argument("--s-offsets", type=str, default=None, help="Comma/space-separated s values for GHD_IT_* files.")
    ap.add_argument(
        "--init-state",
        type=str,
        default=None,
        choices=["neel", "polarized", "neel_even", "vac_fill", "mixed_neel", "vac_infty", "phsymm", "phsymm_odd", "beta", "beta_lr"],
        help="Filter by initial-state family inferred from filename.",
    )
    ap.add_argument("--mat-csv-dir", type=str, default=None,
                help="Directory with Mathematica CSVs (default: <outdir>/GHD_NEEL_MAT_CSV for neel, with GHD_NELL_MAT_CSV kept as a legacy fallback)")
    ap.add_argument("--mat-M", type=int, default=None,
                help="Filter Mathematica files by M (e.g. 800)")
    ap.add_argument(
        "--mat-delta-tag",
        type=str,
        default=None,
        help="Filter hybrid Mathematica CSVs by delta tag (e.g. delta5).",
    )
    ap.add_argument("--z-min", type=float, default=-2.5, help="Minimum zeta shown on x-axis.")
    ap.add_argument("--z-max", type=float, default=2.5, help="Maximum zeta shown on x-axis.")
    ap.add_argument(
        "--x-pad",
        type=int,
        default=DEFAULT_X_PAD,
        help=f"Extra lattice positions added on each side of the x-window beyond r (default: {DEFAULT_X_PAD}).",
    )
    ap.add_argument("--show-title", dest="show_title", action="store_true", help="Show figure title (default: on).")
    ap.add_argument("--no-show-title", dest="show_title", action="store_false", help="Hide figure title.")
    ap.add_argument(
        "--group-mode",
        type=str,
        choices=["r", "r_gamma", "r_gamma_t", "profile_grid"],
        default="r",
        help=(
            "Grouping: 'r' overlays all gamma and T for a fixed r/sign; "
            "'r_gamma' overlays all T for each gamma; "
            "'r_gamma_t' keeps one figure per gamma and T; "
            "'profile_grid' builds a multi-panel overview with one q_x panel per r."
        ),
    )

    ap.set_defaults(show_title=True)
    args = ap.parse_args()
    if args.z_min >= args.z_max:
        raise SystemExit("--z-min must be smaller than --z-max.")
    if args.x_pad < 0:
        raise SystemExit("--x-pad must be non-negative.")

    csv_search_dirs = build_csv_search_dirs(args.outdir, args.py_csv_dir, args.init_state)
    mat_search_dirs = build_mat_search_dirs(args.outdir, args.mat_csv_dir, args.init_state)

    if not csv_search_dirs:
        raise SystemExit(
            "Missing CSV directories. Expected one of: "
            f"{os.path.join(args.outdir, 'GHD_SUPERPOSED', 'csv')} "
            f"(or legacy {os.path.join(args.outdir, 'GHD_SIMP_NELL_CSV')} / {os.path.join(args.outdir, 'GHD_IT_CSV')})."
        )

    r_list = parse_list(args.r_list, int)
    sign_list = parse_list(args.sign, str)
    if sign_list is not None:
        invalid = [s for s in sign_list if s not in {"+", "-"}]
        if invalid:
            raise SystemExit(f"Invalid --sign values: {invalid}. Allowed values are '+' and '-'.")
    gamma_list = parse_list(args.gammas, float)
    size_list = parse_list(args.sizes, int)
    b_offset_list = parse_list(args.b_offsets, int)
    s_offset_list = parse_list(args.s_offsets, int)
    n_list = [2 * s for s in size_list] if size_list else None
    t_list = parse_list(args.times, float)
    if args.time is not None:
        if t_list is None:
            t_list = [args.time]
        else:
            t_list.append(args.time)
    Beta = args.beta
    betaL = args.betaL
    betaR = args.betaR

    py_entries = []
    mat_entries= []

    for csv_dir in csv_search_dirs:
        if not os.path.isdir(csv_dir):
            continue
        for root, _, files in os.walk(csv_dir):
            for fname in files:
                if not fname.endswith(".csv"):
                    continue
                path = os.path.join(root, fname)
                info = parse_name(path)
                if info is None:
                    continue
                if args.init_state is not None and info.get("init_state") != args.init_state:
                    continue
                if Beta is not None:
                    if info.get("beta") is None or not np.isclose(info["beta"], Beta):
                        continue
                if betaL is not None:
                    if info.get("betaL") is None or not np.isclose(info["betaL"], betaL):
                        continue
                if betaR is not None:
                    if info.get("betaR") is None or not np.isclose(info["betaR"], betaR):
                        continue
                if args.phsymm_m is not None:
                    if info.get("m") is None or info["m"] != args.phsymm_m:
                        continue
                if args.phsymm_A is not None:
                    if info.get("A") is None or not np.isclose(info["A"], args.phsymm_A):
                        continue
                if r_list is not None and info["r"] not in r_list:
                    continue
                if sign_list is not None and info["sign"] not in sign_list:
                    continue
                if gamma_list is not None and not any(np.isclose(info["gamma"], g) for g in gamma_list):
                    continue
                if t_list is not None and not any(np.isclose(info["T"], t) for t in t_list):
                    continue
                if n_list is not None and info["N"] not in n_list:
                    continue
                if args.a_offset is not None:
                    if info.get("a") is None or info["a"] != args.a_offset:
                        continue
                if b_offset_list is not None:
                    if info.get("b") is None or info["b"] not in b_offset_list:
                        continue
                if s_offset_list is not None:
                    if info.get("s") is None or info["s"] not in s_offset_list:
                        continue
                py_entries.append((path, info))

    if not py_entries:
        raise SystemExit("No matching CSV files found.")

    if t_list is not None:
        found_times = sorted({float(info["T"]) for _, info in py_entries})
        missing_times = [t for t in t_list if not any(np.isclose(t, ft) for ft in found_times)]
        if missing_times:
            print(
                "WARNING: some requested times were not found after filtering. "
                f"missing={', '.join(fmt2(t) for t in missing_times)} | "
                f"found={', '.join(fmt2(t) for t in found_times)}"
            )
    
    for mat_csv_dir in mat_search_dirs:
        for root, _, files in os.walk(mat_csv_dir):
            for fname in files:
                if not fname.endswith(".csv"):
                    continue
                info = parse_name(fname)
                if info is None or info.get("source") != "mat":
                    continue
                if args.init_state is not None and info.get("init_state") != args.init_state:
                    continue
                if r_list is not None and info["r"] not in r_list:
                    continue
                if sign_list is not None and info["sign"] not in sign_list:
                    continue
                if args.phsymm_m is not None:
                    if info.get("m") is None or info["m"] != args.phsymm_m:
                        continue
                if args.phsymm_A is not None:
                    if info.get("A") is None or not np.isclose(info["A"], args.phsymm_A):
                        continue
                if gamma_list is not None and not any(np.isclose(info["gamma"], g) for g in gamma_list):
                    continue
                if args.mat_M is not None:
                    if info.get("M") is None or info["M"] != args.mat_M:
                        continue
                if args.mat_delta_tag is not None:
                    if info.get("delta_tag") != args.mat_delta_tag:
                        continue
                mat_entries.append((os.path.join(root, fname), info))


    # Group Python entries into figures.
    py_groups = {}
    for path, info in py_entries:
        key = (
            info.get("beta"),
            info.get("betaL"),
            info.get("betaR"),
            info.get("init_state"),
            info["r"],
            info["sign"],
            info["N"],
            info.get("a"),
            info.get("b"),
            info.get("s"),
            info.get("m"),
            info.get("A"),
        )
        if args.group_mode in ("r_gamma", "r_gamma_t"):
            key = key + (info["gamma"],)
        if args.group_mode == "r_gamma_t":
            key = key + (info["T"],)
        py_groups.setdefault(key, []).append((path, info))

    # Mathematica lookup
    mat_groups = {}
    for path, info in mat_entries:
        if info.get("init_state") in ("phsymm", "phsymm_odd"):
            key = (info.get("init_state"), info["r"], info["sign"], info["gamma"], info.get("m"), info.get("A"))
        else:
            key = (info.get("init_state"), info["r"], info["sign"], info["gamma"])
        mat_groups.setdefault(key, []).append((path, info))

    if args.mat_delta_tag is None:
        ambiguous_keys = []
        for key, items in mat_groups.items():
            delta_tags = sorted({info.get("delta_tag") for _, info in items if info.get("delta_tag")})
            if len(delta_tags) > 1:
                ambiguous_keys.append((key, delta_tags))
        if ambiguous_keys:
            preview = []
            for key, delta_tags in ambiguous_keys[:5]:
                preview.append(
                    f"{key[0]}, r={key[1]}, sign={key[2]}, gamma={fmt2(key[3])}: {', '.join(delta_tags)}"
                )
            raise SystemExit(
                "Multiple Mathematica delta tags matched the same state/r/sign/gamma key. "
                "Pass --mat-delta-tag (for example delta5). "
                f"Conflicts: {'; '.join(preview)}"
            )

    mat_tag_suffix = f"_{args.mat_delta_tag}" if args.mat_delta_tag else ""

    png_root_dir = os.path.join(args.outdir, "GHD_SUPERPOSED", "png")
    pdf_root_dir = os.path.join(args.outdir, "GHD_SUPERPOSED", "pdf")
    os.makedirs(png_root_dir, exist_ok=True)
    os.makedirs(pdf_root_dir, exist_ok=True)

    def _parse_num_token(tok):
        s = str(tok).strip().strip('"').strip("'")
        if s == "":
            return np.nan
        try:
            return float(s)
        except ValueError:
            # Mathematica may export rationals like -19/5 in CSV.
            if "/" in s:
                try:
                    n, d = s.split("/", 1)
                    return float(n) / float(d)
                except ValueError:
                    return np.nan
            return np.nan

    def load_profile(path, source):
        if source == "py":
            data = np.loadtxt(path,  delimiter=",", skiprows=1)
            return data[:,2], data[:,3], data[:,4] # zeta, q , J
        else:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[1] < 3:
                raise ValueError(f"Unexpected Mathematica CSV shape in {path}: {raw.shape}")
            rows = []
            for row in raw:
                vals = [_parse_num_token(row[i]) for i in range(3)]
                if np.isfinite(vals).all():
                    rows.append(vals)
            data = np.asarray(rows, dtype=float)
            if data.size == 0:
                raise ValueError(f"No numeric rows found in Mathematica CSV: {path}")
            return data[:,0], data[:,1], data[:,2] # zeta, q , J

    def cross_boundary_x_vals(r_val, n_val, x_pad):
        center = int(n_val) // 2
        r_val = r_val + int(x_pad)
        return np.arange(center - int(r_val) + 1, center + + int(r_val)+1, dtype=int)

    def sample_qx_from_profile(zeta_axis, q_axis, r_val, t_val, n_val):
        center = int(n_val) // 2
        x_targets = cross_boundary_x_vals(r_val, n_val, args.x_pad)
        zeta_targets = (x_targets - (center - 1)) / float(t_val)
        order = np.argsort(zeta_axis)
        zeta_sorted = np.asarray(zeta_axis, dtype=float)[order]
        q_sorted = np.asarray(q_axis, dtype=float)[order]
        q_x = np.interp(zeta_targets, zeta_sorted, q_sorted)
        x_rel = x_targets - (center - 1)
        return x_rel, q_x

    def qx_ylabel(sign_val):
        if sign_val == "-":
            return r"$2J\,\mathrm{Im}[C_{x,x+r}(t)]$"
        return r"$2J\,\mathrm{Re}[C_{x,x+r}(t)]$"


    markers = ["o", "s", "^"]
    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:brown",
        "tab:cyan",
        "tab:pink",
        "tab:gray",
    ]

    def sort_group_items(items):
        items = list(items)
        items.sort(key=lambda x: (
            x[1]["gamma"],
            x[1]["T"],
            x[1]["b"] if x[1].get("b") is not None else 10**9,
            x[1]["s"] if x[1].get("s") is not None else 10**9,
        ))
        return items

    def format_gamma_meta(gamma_vals):
        if len(gamma_vals) == 1:
            return rf"\gamma={gamma_vals[0]:g}", f"gamma{gamma_vals[0]:g}"
        return r"\gamma\in\{" + ", ".join(f"{g:g}" for g in gamma_vals) + r"\}", f"g{len(gamma_vals)}"

    def format_time_meta(t_vals):
        if len(t_vals) == 1:
            return rf"T={fmt2(t_vals[0])}", f"T{fmt2(t_vals[0])}"
        return r"T\in\{" + ", ".join(fmt2(t) for t in t_vals) + r"\}", f"T{fmt2(t_vals[0])}_to_{fmt2(t_vals[-1])}_{len(t_vals)}pts"

    def format_n_meta(n_vals):
        n_vals = sorted(int(n) for n in n_vals)
        if len(n_vals) == 1:
            return str(n_vals[0]), f"N{n_vals[0]}"
        return r"\{" + ", ".join(str(n) for n in n_vals) + r"\}", "N" + "_".join(str(n) for n in n_vals)

    def build_group_meta(items):
        items = sort_group_items(items)
        info0 = items[0][1]
        gamma_vals = sorted({float(it[1]["gamma"]) for it in items})
        t_vals = sorted({float(it[1]["T"]) for it in items})
        n_vals = sorted({int(it[1]["N"]) for it in items})
        gamma_txt, gamma_tag = format_gamma_meta(gamma_vals)
        t_txt, t_tag = format_time_meta(t_vals)
        include_gamma_in_label = (len(gamma_vals) > 1) or (len(t_vals) == 1)
        include_time_in_label = (len(t_vals) > 1) or (len(gamma_vals) == 1)
        beta_val = info0.get("beta")
        init_state = info0.get("init_state")
        if init_state == "neel":
            title = rf"$r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "polarized":
            title = rf"$\mathrm{{Polarized}},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "neel_even":
            title = rf"$\mathrm{{Vacuum\ Neel\ even}},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "vac_fill":
            title = rf"$\mathrm{{Vac/Fill}},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "mixed_neel":
            title = rf"$\mathrm{{Mixed\ Neel}},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "vac_infty":
            title = rf"$\mathrm{{Vac/Infty}},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "phsymm":
            title = rf"$\mathrm{{PHSYMM}},\ m={info0['m']},\ A={info0['A']:.6g},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "phsymm_odd":
            title = rf"$\mathrm{{PHSYMM\ odd}},\ m={info0['m']},\ A={info0['A']:.6g},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        elif init_state == "beta_lr":
            title = rf"$\beta_L={info0['betaL']},\ \beta_R={info0['betaR']},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        else:
            title = rf"$\beta={beta_val},\ r={info0['r']},\ \mathrm{{sign}}={info0['sign']},\ {gamma_txt},\ {t_txt},\ N={info0['N']}$"
        return {
            "items": items,
            "info0": info0,
            "beta_val": beta_val,
            "init_state": init_state,
            "r_val": info0["r"],
            "sign_val": info0["sign"],
            "gamma_vals": gamma_vals,
            "t_vals": t_vals,
            "n_vals": n_vals,
            "gamma_txt": gamma_txt,
            "gamma_tag": gamma_tag,
            "t_txt": t_txt,
            "t_tag": t_tag,
            "include_gamma_in_label": include_gamma_in_label,
            "include_time_in_label": include_time_in_label,
            "include_n_in_label": False,
            "title": title,
            "region_tag": format_region_tag(info0),
            "state_tag": init_state if init_state in INIT_STATES else "unknown",
        }

    def find_mat_profile(meta, info, zeta):
        init_state = meta["init_state"]
        g_key = float(info["gamma"])
        if init_state in ("phsymm", "phsymm_odd"):
            mat_key = (init_state, meta["r_val"], meta["sign_val"], g_key, info.get("m"), info.get("A"))
        else:
            mat_key = (init_state, meta["r_val"], meta["sign_val"], g_key)
        mat_items = mat_groups.get(mat_key, [])
        if not mat_items and init_state == "neel":
            mat_items = mat_groups.get((None, meta["r_val"], meta["sign_val"], g_key), [])
        if not mat_items and init_state == "vac_fill" and np.isclose(g_key, 0.0):
            mat_dir_for_save = args.mat_csv_dir or os.path.join(args.outdir, "GHD_vac_fill_mat", "csv")
            mat_path = ensure_vac_fill_unitary_csv(mat_dir_for_save, meta["r_val"], meta["sign_val"], zeta)
            mat_info = {
                "source": "mat",
                "init_state": "vac_fill",
                "r": meta["r_val"],
                "sign": meta["sign_val"],
                "gamma": 0.0,
                "M": 0,
            }
            mat_items = [(mat_path, mat_info)]
        if not mat_items:
            return None
        mat_items = sorted(
            mat_items,
            key=lambda x: (
                x[1].get("M", -1),
                1 if "seggrid" in str(x[1].get("tag", "")).lower() else 0,
            ),
        )
        for cand_path, cand_info in reversed(mat_items):
            try:
                zeta_m, q_m, j_m = load_profile(cand_path, "mat")
                return zeta_m, q_m, j_m, cand_info
            except ValueError:
                continue
        return None

    def draw_group(meta, ax, label_prefix=None, color_offset=0, legend_mode="both"):
        init_state = meta["init_state"]
        x_vals_ref = cross_boundary_x_vals(meta["r_val"], meta["info0"]["N"], args.x_pad) - (meta["info0"]["N"] // 2)
        for idx, (path, info) in enumerate(meta["items"]):
            zeta, q_vals, _ = load_profile(path, "py")
            x_vals, q_x = sample_qx_from_profile(zeta, q_vals, meta["r_val"], info["T"], info["N"])
            color = color_cycle[(color_offset + idx) % len(color_cycle)]
            label_parts = []
            if label_prefix is not None:
                label_parts.append(label_prefix)
            elif idx == 0:
                label_parts.append(init_state_tex_label(init_state))
            if label_prefix is not None and idx == 0:
                label_parts.append(init_state_tex_label(init_state))
            if meta["include_gamma_in_label"]:
                label_parts.append(rf"\gamma={info['gamma']:g}")
            if meta["include_time_in_label"]:
                label_parts.append(rf"T={fmt2(info['T'])}")
            if meta.get("include_n_in_label", False):
                label_parts.append(rf"N={info['N']}")
            label = "$" + r",\ ".join(label_parts) + "$"
            marker = markers[idx % len(markers)]
            ax.scatter(x_vals, q_x, s=26, marker=marker, color=color, alpha=0.55, linewidths=0, label=label)

        ax.set_xlim(x_vals_ref[0] - 0.3, x_vals_ref[-1] + 0.3)
        ax.grid(True, ls="--", alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.6g"))
        ax.set_ylabel(qx_ylabel(meta["sign_val"]), labelpad=6)
        if legend_mode == "both":
            ax.legend(fontsize=11, loc="best")
        elif legend_mode == "q_only":
            ax.legend(fontsize=9, loc="best")

    def build_single_output_paths(meta):
        png_dir = os.path.join(png_root_dir, meta["state_tag"])
        pdf_dir = os.path.join(pdf_root_dir, meta["state_tag"])
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)
        init_state = meta["init_state"]
        info = meta["info0"]
        if init_state == "neel":
            outpng = os.path.join(png_dir, f"GHD_NEEL_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "polarized":
            outpng = os.path.join(png_dir, f"GHD_POLARIZED_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "neel_even":
            outpng = os.path.join(png_dir, f"GHD_IT_NEEL_EVEN_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "vac_fill":
            outpng = os.path.join(png_dir, f"GHD_vac_fill_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "mixed_neel":
            outpng = os.path.join(png_dir, f"GHD_IT_MixedNeel_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "vac_infty":
            outpng = os.path.join(png_dir, f"GHD_IT_VacInfty_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "phsymm":
            outpng = os.path.join(png_dir, f"GHD_IT_PHSYMM_m{info['m']}_A{info['A']:.6g}_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "phsymm_odd":
            outpng = os.path.join(png_dir, f"GHD_IT_PHSYMM_ODD_m{info['m']}_A{info['A']:.6g}_xsuperposed{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        elif init_state == "beta_lr":
            outpng = os.path.join(png_dir, f"GHD_NEEL_xsuperposed_betaL{info['betaL']}_betaR{info['betaR']}{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        else:
            outpng = os.path.join(png_dir, f"GHD_NEEL_xsuperposed_Beta{meta['beta_val']}{meta['region_tag']}_r{meta['r_val']}_sign{meta['sign_val']}_{meta['gamma_tag']}_{meta['t_tag']}_N{info['N']}{mat_tag_suffix}.png")
        return outpng, os.path.join(pdf_dir, os.path.basename(outpng).replace(".png", ".pdf"))

    group_metas = [build_group_meta(items) for items in py_groups.values()]

    if args.group_mode == "profile_grid":
        r_order = r_list if r_list is not None else sorted({m["r_val"] for m in group_metas})
        n_profiles = len(r_order)
        if n_profiles == 0:
            raise SystemExit("No groups available for --group-mode profile_grid.")
        rows = 1
        cols = n_profiles
        fig_width = max(5.2, 2.8 * cols)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, 3.9), sharex="col", squeeze=False)
        used_metas = []
        for col, r_val in enumerate(r_order):
            q_ax = axes[0, col]
            col_metas = sorted(
                [m for m in group_metas if m["r_val"] == r_val],
                key=lambda m: (0 if m["sign_val"] == "-" else 1, m["sign_val"], m["region_tag"]),
            )
            if not col_metas:
                q_ax.set_title(rf"$r={r_val}$", fontsize=13, pad=8)
                q_ax.text(
                    0.5, 0.5, "no data",
                    transform=q_ax.transAxes,
                    ha="center", va="center", fontsize=11, alpha=0.7,
                )
                continue

            multi_meta = len(col_metas) > 1
            multi_n = len({meta["info0"]["N"] for meta in col_metas}) > 1
            for idx_meta, meta in enumerate(col_metas):
                meta["include_n_in_label"] = multi_n
                label_prefix = rf"\mathrm{{sign}}={meta['sign_val']}" if multi_meta else None
                draw_group(meta, q_ax, label_prefix=label_prefix, color_offset=3 * idx_meta, legend_mode="none")
                used_metas.append(meta)

            q_ax.set_title(rf"$r={r_val}$", fontsize=13, pad=8)
            handles, labels = q_ax.get_legend_handles_labels()
            if handles:
                q_ax.legend(fontsize=8.5, loc="best")

        for col in range(cols):
            axes[0, col].set_xlabel(r"$x$", labelpad=8)
            axes[0, col].yaxis.tick_left()
            if col == 0:
                axes[0, col].yaxis.set_label_coords(-0.23, 0.5)
            else:
                axes[0, col].set_ylabel("")
                axes[0, col].tick_params(labelleft=True)

        if not used_metas:
            plt.close(fig)
            raise SystemExit("No matching data found for the requested r values in --group-mode profile_grid.")

        ref_meta = used_metas[0]
        gamma_vals = sorted({g for meta in used_metas for g in meta["gamma_vals"]})
        t_vals = sorted({t for meta in used_metas for t in meta["t_vals"]})
        gamma_txt, gamma_tag = format_gamma_meta(gamma_vals)
        t_txt, t_tag = format_time_meta(t_vals)
        n_vals = sorted({meta["info0"]["N"] for meta in used_metas})
        n_txt, n_tag = format_n_meta(n_vals)
        if args.show_title:
            title = rf"${init_state_panel_label(ref_meta['init_state'])},\ {gamma_txt},\ {t_txt},\ N={n_txt}$"
            fig.suptitle(title, fontsize=17, y=0.995)
            fig.subplots_adjust(left=0.075, right=0.995, bottom=0.08, top=0.90, wspace=0.30, hspace=0.14)
        else:
            fig.subplots_adjust(left=0.075, right=0.995, bottom=0.08, top=0.96, wspace=0.30, hspace=0.14)

        profile_tag = "_".join(
            [f"r{r}" for r in r_order]
        )
        pdf_dir = os.path.join(pdf_root_dir, ref_meta["state_tag"])
        os.makedirs(pdf_dir, exist_ok=True)
        outpdf = os.path.join(
            pdf_dir,
            f"{ref_meta['init_state']}_x_profile_grid_{profile_tag}_{gamma_tag}_{t_tag}_{n_tag}_{rows}x{cols}{mat_tag_suffix}.pdf",
        )
        fig.savefig(outpdf, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"wrote {outpdf}")
    else:
        for meta in group_metas:
            fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), sharex=True)
            draw_group(meta, ax, legend_mode="both")
            ax.set_xlabel(r"$x$", labelpad=8)
            if args.show_title:
                fig.suptitle(meta["title"], fontsize=16, y=0.98)
                fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.86)
            else:
                fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)
            outpng, outpdf = build_single_output_paths(meta)
            fig.savefig(outpng, dpi=220, bbox_inches="tight", pad_inches=0.08)
            fig.savefig(outpdf, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            print(f"wrote {outpng}")
            print(f"wrote {outpdf}")


if __name__ == "__main__":
    main()
