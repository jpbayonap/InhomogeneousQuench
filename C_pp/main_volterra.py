import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless (safe on servers)
import matplotlib.pyplot as plt
from scipy.special import jv
from pathlib import Path

# ----------------------------
# Inputs: a_-(t) and g_-(t)
# ----------------------------

def a_minus_halfneel(t, J=1.0, nmax_extra=40):
    """
    Half-vacuum / half-Néel (site 0 empty, occupied odd sites j>=1), infinite chain.
    For this benchmark:
        a_+(t)=0,
        a_-(t)= 2*i * sum_{m>=0} J_{2m}(z) J_{2m+1}(z), z = 2Jt.
    Truncate Bessel orders at nmax ~ zmax + extra.
    """
    z = 2.0 * J * t
    zmax = float(np.max(z))
    nmax = int(np.ceil(zmax + nmax_extra)) + 2

    orders = np.arange(nmax + 1, dtype=int)[:, None]  # (n,1)
    Jnz = jv(orders, z[None, :])                      # (n,Nt)

    Je = Jnz[0::2, :]  # J_0, J_2, ...
    Jo = Jnz[1::2, :]  # J_1, J_3, ...

    M = min(Je.shape[0], Jo.shape[0])
    return 2.0j * np.sum(Je[:M, :] * Jo[:M, :], axis=0)

def g_minus_kernel(t, J=1.0):
    z = 2.0 * J * t
    return jv(0, z)**2 - jv(1, z)**2


# ----------------------------
# Volterra solver (trapezoid)
# ----------------------------

def solve_volterra_scalar_trap(t, a, g, gamma):
    """
    Solve: c(t) = a(t) - gamma * ∫_0^t g(t-s) c(s) ds
    on uniform grid t[i]=i*dt, trapezoidal rule.
    """
    dt = t[1] - t[0]
    N = len(t) - 1
    c = np.zeros_like(a, dtype=np.complex128)
    c[0] = a[0]

    denom = 1.0 + 0.5 * gamma * dt * g[0]  # g(0)=1 here

    for i in range(1, N + 1):
        conv_sum = 0.0 + 0.0j if i <= 1 else np.dot(g[1:i], c[i-1:0:-1])
        rhs = a[i] - gamma * dt * (conv_sum + 0.5 * g[i] * c[0])
        c[i] = rhs / denom

    return c


# ----------------------------
# Your corrected asymptote
# ----------------------------

def cminus_infty_corrected(gamma):
    return 1j * (1.0 ) / (np.pi + gamma)


# ----------------------------
# Run + plot + save CSV
# ----------------------------

def run_case(J=1.0, gamma=0.1, tmax=300.0, dt=0.1, outdir="output"):
    t = np.arange(0.0, tmax + 1e-12, dt)

    a = a_minus_halfneel(t, J=J, nmax_extra=40)
    g = g_minus_kernel(t, J=J)
    c = solve_volterra_scalar_trap(t, a, g, gamma)

    cinf = cminus_infty_corrected(gamma)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"cminus_corrected_J{J}_g{gamma}_tmax{tmax}_dt{dt}.csv"
    pd.DataFrame({"t": t, "Re_cminus": np.real(c), "Im_cminus": np.imag(c)}).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, np.real(c), label="Re c_-(t)")
    ax.plot(t, np.imag(c), label="Im c_-(t)")
    ax.axhline(np.real(cinf), ls="--", label=rf"$Re c_-(\infty) corrected$")
    ax.axhline(np.imag(cinf), ls="--", label=rf"Im c_-(\infty) corrected")
    ax.set_xlabel("t")
    ax.set_ylabel("c_-(t)")
    ax.set_title(rf"Volterra: c_-(t) vs corrected asymptote (J={J}, \gamma={gamma})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    img_path = outdir / f"cminus_corrected_J{J}_g{gamma}_tmax{tmax}_dt{dt}.png"
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    print("Saved plot:", img_path)
    print("Saved csv :", csv_path)
    print("c_-(∞) corrected =", cinf)
    print("c_-(tmax)        =", c[-1])

    return img_path, csv_path


if __name__ == "__main__":
    # weak-gamma sweep
    run_case(J=1.0, gamma=0.10, tmax=300.0, dt=0.1, outdir="output")
    run_case(J=1.0, gamma=1.0, tmax=600.0, dt=0.1, outdir="output")
