#!/usr/bin/env python3
"""
Utility to compare two GHD CSV outputs (chi,q,J).

Usage:
  python compare_profiles.py --ref path/to/mathematica.csv --test path/to/cpp.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def load_three_columns(path: Path) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                z, q, J = map(float, parts)
            except ValueError:
                continue
            rows.append((z, q, J))
    return rows
 

def compare(ref: List[Tuple[float, float, float]], test: List[Tuple[float, float, float]]):
    n = min(len(ref), len(test))
    if n == 0:
        return None
    max_dq = max(abs(test[i][1] - ref[i][1]) for i in range(n))
    max_dJ = max(abs(test[i][2] - ref[i][2]) for i in range(n))
    return n, max_dq, max_dJ


def main():
    parser = argparse.ArgumentParser(description="Compare two GHD CSV files (zeta,q,J).")
    parser.add_argument("--ref", required=True, type=Path, help="Reference CSV (e.g., Mathematica, Python).")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV (e.g., C++ output).")
    parser.add_argument("--samples", type=int, default=4, help="Number of evenly spaced sample rows to print.")
    args = parser.parse_args()

    if not args.ref.exists():
        raise SystemExit(f"Reference file not found: {args.ref}")
    if not args.test.exists():
        raise SystemExit(f"Test file not found: {args.test}")

    ref_rows = load_three_columns(args.ref)
    test_rows = load_three_columns(args.test)

    if not ref_rows or not test_rows:
        raise SystemExit("One of the files has no readable rows.")

    n = min(len(ref_rows), len(test_rows))
    res = compare(ref_rows, test_rows)
    if res is None:
        raise SystemExit("No rows to compare.")

    n_rows, max_dq, max_dJ = res
    print(f"Ref rows: {len(ref_rows)} Test rows: {len(test_rows)} Compared: {n_rows}")
    print(f"max |Δq| = {max_dq:.6g}")
    print(f"max |ΔJ| = {max_dJ:.6g}")

    if args.samples > 0:
        step = max(1, n_rows // args.samples)
        print("\nSamples (index: z_ref, q_ref, J_ref | z_test, q_test, J_test)")
        for idx in range(0, n_rows, step):
            zr, qr, Jr = ref_rows[idx]
            zt, qt, Jt = test_rows[idx]
            print(f"{idx:4d}: {zr:.6f}, {qr:.6f}, {Jr:.6f} | {zt:.6f}, {qt:.6f}, {Jt:.6f}")


if __name__ == "__main__":
    main()
