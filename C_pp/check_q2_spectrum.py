#!/usr/bin/env python3
"""
Compare the spectrum of the real-space Q2 operator against 2 cos(hop*k)
for both OBC and PBC.

Q2 is defined as a symmetric hopping matrix with distance = hop:
  Q2_{i,i+hop} = Q2_{i+hop,i} = 1 (with optional wrap for PBC).

OBC analytic k: k_n = pi * n / (L + 2), n=1..L  (matches sin(2k(j+1)) convention)
PBC analytic k: k_n = 2 pi * n / L, n=0..L-1

Run:
  python3 check_q2_spectrum.py --L 200 --hop 2 --plot
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_q2_matrix(L: int, hop: int, bc: str) -> np.ndarray:
    Q = np.zeros((L, L), dtype=float)
    for i in range(L):
        j = i + hop
        if bc == "pbc":
            j %= L
            Q[i, j] += 1.0
            Q[j, i] += 1.0
        else:
            if j < L:
                Q[i, j] += 1.0
                Q[j, i] += 1.0
    return Q


def analytic_eigs_obc(L: int, hop: int) -> np.ndarray:
    n = np.arange(1, L + 1)
    k = np.pi * n / (L + 2.0)
    return 2.0 * np.cos(hop * k)


def analytic_eigs_pbc(L: int, hop: int) -> np.ndarray:
    n = np.arange(0, L)
    k = 2.0 * np.pi * n / L
    return 2.0 * np.cos(hop * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--hop", type=int, default=2, help="hopping distance; hop=2 gives 2 cos(2k)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    for bc in ("obc", "pbc"):
        Q = build_q2_matrix(args.L, args.hop, bc)
        eigs_num = np.linalg.eigvalsh(Q)
        if bc == "obc":
            eigs_ana = analytic_eigs_obc(args.L, args.hop)
        else:
            eigs_ana = analytic_eigs_pbc(args.L, args.hop)

        eigs_num = np.sort(eigs_num)
        eigs_ana = np.sort(eigs_ana)
        diff = eigs_num - eigs_ana
        print(f"[{bc}] max|diff|={np.max(np.abs(diff)):.3e}, mean|diff|={np.mean(np.abs(diff)):.3e}")

        if args.plot:
            plt.figure(figsize=(7, 4))
            plt.plot(eigs_num, "o", ms=3, label="numeric")
            plt.plot(eigs_ana, "-", lw=2, label="analytic")
            plt.title(f"{bc.upper()} spectrum: L={args.L}, hop={args.hop}")
            plt.xlabel("index (sorted)")
            plt.ylabel("eigenvalue")
            plt.grid(True, ls="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            out = f"q2_spectrum_{bc}_L{args.L}_hop{args.hop}.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
