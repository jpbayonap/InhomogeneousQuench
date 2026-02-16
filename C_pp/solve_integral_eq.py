#!/usr/bin/env python3
"""
Solve the truncated linear system for the coefficients a^R_n, a^L_n, b^R_n, b^L_n
appearing in the integral equation

 sum_n [ F^{(n)}(r) (a^R_n+(-1)^n a^L_n)
        + G^{(n)}_r (b^R_n+(-1)^n b^L_n)
        + (1-(-1)^{r+n})/(n^2-r^2) ( n[b^R_n-(-1)^n b^L_n] - r[a^R_n-(-1)^n a^L_n] )(1-δ_{r,n}) ]
   = M(r,beta_L)-M(r,beta_R) + N(r,beta_L)+N(r,beta_R)
     + (J r γ π /2) ( a^R_r+(-1)^r a^L_r + b^R_r+(-1)^r b^L_r ).

We form a dense linear system for the symmetric/antisymmetric combinations:
  s_a[n] = a^R_n + (-1)^n a^L_n
  d_a[n] = a^R_n - (-1)^n a^L_n
  s_b[n] = b^R_n + (-1)^n b^L_n
  d_b[n] = b^R_n - (-1)^n b^L_n

Unknown vector x = [s_a, s_b, d_a, d_b] has length 4M. The system gives M equations,
so we solve it in the least-squares sense with Tikhonov regularization (lambda_reg).
If you want a square system, you can set d_a=d_b=0 by choosing `use_only_sym=True`.

Usage example:
    python3 solve_integral_eq.py --M 50 --betaL 0.5 --betaR 1.0 --gamma 0.1 --J 1.0
"""

import argparse
import numpy as np
from scipy.integrate import quad

def F_kernel(n, r, J):
    if r == n:
        return -16 * J * J * (n * n) / (4 * n * n - 1)
    denom = n**4 - 2 * n * n * (r * r + 1) + (r * r - 1) ** 2
    return 8 * J * J * (r * n * (1 + (-1) ** (r + n))) / denom


def G_kernel(n, r, J):
    if r == n:
        return 16 * J * J * (n * n) / (4 * n * n - 1)
    denom = n**4 - 2 * n * n * (r * r + 1) + (r * r - 1) ** 2
    return 4 * J * J * ((n * n + r * r - 1) * (1 + (-1) ** (r + n))) / denom


def M_term(r, beta, J):
    integrand = lambda k: np.sin(r * k) / (np.exp(-2 * J * beta * np.cos(k)) + 1.0)
    val, _ = quad(integrand, 0, np.pi, limit=400)
    return val


def N_term(r, beta, J):
    integrand = lambda k: np.cos(r * k) / (np.exp(-2 * J * beta * np.cos(k)) + 1.0)
    val, _ = quad(integrand, 0, np.pi, limit=400)
    return val


def build_system(M, betaL, betaR, gamma, J, use_only_sym=False, lambda_reg=0.0):
    """
    Returns A, b where A x = b. If use_only_sym is True, we only solve for s_a,s_b (length 2M)
    assuming d_a=d_b=0.
    """
    F = np.zeros((M, M))
    G = np.zeros((M, M))
    for r in range(1, M + 1):
        for n in range(1, M + 1):
            F[r - 1, n - 1] = F_kernel(n, r, J)
            G[r - 1, n - 1] = G_kernel(n, r, J)

    rhs = np.zeros(M)
    for r in range(1, M + 1):
        rhs[r - 1] = (M_term(r, betaL, J) - M_term(r, betaR, J) +
                      N_term(r, betaL, J) + N_term(r, betaR, J))

    if use_only_sym:
        # Unknowns: [s_a, s_b], length 2M
        A = np.zeros((M, 2 * M))
        for r in range(1, M + 1):
            row = r - 1
            for n in range(1, M + 1):
                A[row, n - 1] += F[row, n - 1]
                A[row, M + n - 1] += G[row, n - 1]
            # add gamma term on diagonal for r
            A[row, r - 1] += 0.5 * J * r * gamma * np.pi
            A[row, M + r - 1] += 0.5 * J * r * gamma * np.pi
        if lambda_reg > 0:
            A_reg = np.vstack([A, np.sqrt(lambda_reg) * np.eye(2 * M)])
            b_reg = np.concatenate([rhs, np.zeros(2 * M)])
            sol, *_ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
        else:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        s_a = sol[:M]
        s_b = sol[M:]
        d_a = np.zeros(M)
        d_b = np.zeros(M)
        return s_a, s_b, d_a, d_b, A, rhs

    # Full least-squares for [s_a, s_b, d_a, d_b]
    A = np.zeros((M, 4 * M))
    for r in range(1, M + 1):
        row = r - 1
        for n in range(1, M + 1):
            parity_factor = 1 - (-1) ** (r + n)
            if r != n:
                coeff = parity_factor / (n * n - r * r)
                A[row, 2 * M + n - 1] += -r * coeff    # d_a_n
                A[row, 3 * M + n - 1] += n * coeff     # d_b_n
            A[row, n - 1] += F[row, n - 1]            # s_a_n
            A[row, M + n - 1] += G[row, n - 1]        # s_b_n
        # gamma term on r for s_a_r and s_b_r
        A[row, r - 1] += 0.5 * J * r * gamma * np.pi
        A[row, M + r - 1] += 0.5 * J * r * gamma * np.pi

    if lambda_reg > 0:
        A_reg = np.vstack([A, np.sqrt(lambda_reg) * np.eye(4 * M)])
        b_reg = np.concatenate([rhs, np.zeros(4 * M)])
        sol, *_ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    else:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    s_a = sol[:M]
    s_b = sol[M:2 * M]
    d_a = sol[2 * M:3 * M]
    d_b = sol[3 * M:]
    return s_a, s_b, d_a, d_b, A, rhs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50, help="truncation")
    ap.add_argument("--betaL", type=float, default=0.5)
    ap.add_argument("--betaR", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--only-sym", action="store_true", help="set d_a=d_b=0 and solve reduced system")
    ap.add_argument("--lambda-reg", type=float, default=0.0, help="Tikhonov regularization")
    args = ap.parse_args()

    s_a, s_b, d_a, d_b, A, rhs = build_system(
        args.M, args.betaL, args.betaR, args.gamma, args.J,
        use_only_sym=args.only_sym, lambda_reg=args.lambda_reg
    )

    res = A @ np.concatenate([s_a, s_b, d_a, d_b]) - rhs if not args.only_sym else \
          A @ np.concatenate([s_a, s_b]) - rhs
    print(f"Residual norm: {np.linalg.norm(res):.3e}")
    print("s_a (first 5):", s_a[:5])
    print("s_b (first 5):", s_b[:5])
    if not args.only_sym:
        print("d_a (first 5):", d_a[:5])
        print("d_b (first 5):", d_b[:5])


if __name__ == "__main__":
    main()
