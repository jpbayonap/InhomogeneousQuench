// dynamics_module.hpp
// Minimal C++ translation of the sparse helpers in dynamics_module.py.
// Uses Eigen sparse matrices and a small Krylov expm_multiply analogue
// suitable for applying exp(t*A) to a vector without forming dense matrices.

#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <complex>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace dyn
{

    using cplx = std::complex<double>;
    using SpMat = Eigen::SparseMatrix<cplx>;
    using SpMatReal = Eigen::SparseMatrix<double>;
    using Vec = Eigen::VectorXcd;
    using Mat = Eigen::MatrixXcd;
    using RowMat = Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // ---------- Hamiltonian ----------
    // Tridiagonal hopping with optional periodic bonds. Size N=2L.
    inline SpMat Hamiltonian(int L, double t1, double mu = 0.0, const std::string &bc = "open")
    {
        const int N = 2 * L;
        std::vector<Eigen::Triplet<cplx>> triplets;
        triplets.reserve(3 * N);

        for (int i = 0; i < N; ++i)
        {
            triplets.emplace_back(i, i, cplx{mu, 0.0});
        }
        for (int i = 0; i < N - 1; ++i)
        {
            triplets.emplace_back(i, i + 1, cplx{-t1, 0.0});
            triplets.emplace_back(i + 1, i, cplx{-t1, 0.0});
        }
        if (bc == "periodic" || bc == "pbc" || bc == "Periodic" || bc == "PBC")
        {
            triplets.emplace_back(0, N - 1, cplx{t1, 0.0});
            triplets.emplace_back(N - 1, 0, cplx{t1, 0.0});
        }
        // Sparse implementation for the free Hamiltonian
        SpMat H(N, N);
        H.setFromTriplets(triplets.begin(), triplets.end());
        return H;
    }

    // ---------- Inhomogeneous initial state ----------
    // Matches Gamma_0 in the Python: vacuum on the left, Néel pattern on the right.
    inline Mat Gamma0(int L)
    {
        const int N = 2 * L;
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(N);
        for (int i = 0; i < L; ++i)
        {
            if (i % 2 == 1)
            {
                diag(L + i) = 1.0; // Occupancy on odd sites of the right half
            }
        }

        return diag.cast<cplx>().asDiagonal();
    }

    // ---------- Vectorization helpers (row-major / C order) ----------
    inline Vec mat2vec(const Mat &C)
    {
        // Row-stacking (C order): vec(C)[i*N + j] = C(i,j)
        RowMat tmp = C;
        return Eigen::Map<Vec>(tmp.data(), tmp.size());
    }

    inline Mat vec2mat(const Vec &v, int N)
    {
        // Inverse of mat2vec for row-major layout
        RowMat tmp = Eigen::Map<const RowMat>(v.data(), N, N);
        return Mat(tmp);
    }

    // ---------- Kronecker-like superoperators (row-stacking convention) ----------
    inline SpMat spre_sp(const SpMat &A)
    {
        // (A ⊗ I) in row-stacked convention: idx(i,j) = i*N + j
        const int N = static_cast<int>(A.rows());
        std::vector<Eigen::Triplet<cplx>> triplets;
        triplets.reserve(static_cast<std::size_t>(A.nonZeros()) * static_cast<std::size_t>(N));

        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (SpMat::InnerIterator it(A, k); it; ++it)
            {
                const int i = static_cast<int>(it.row());
                const int j = static_cast<int>(it.col());
                const cplx val = it.value();
                for (int l = 0; l < N; ++l)
                {
                    // row = idx(i, l) = i*N + l, col = idx(j, l) = j*N + l
                    const int row = i * N + l;
                    const int col = j * N + l;
                    triplets.emplace_back(row, col, val);
                }
            }
        }

        SpMat K(N * N, N * N);
        K.setFromTriplets(triplets.begin(), triplets.end());
        return K;
    }

    inline SpMat spost_sp(const SpMat &A)
    {
        // (I ⊗ A^T) in row-stacked convention: idx(i,j) = i*N + j
        const int N = static_cast<int>(A.rows());
        std::vector<Eigen::Triplet<cplx>> triplets;
        triplets.reserve(static_cast<std::size_t>(A.nonZeros()) * static_cast<std::size_t>(N));

        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (SpMat::InnerIterator it(A, k); it; ++it)
            {
                const int a_row = static_cast<int>(it.row()); // corresponds to j in C*A
                const int a_col = static_cast<int>(it.col()); // corresponds to l in C*A
                const cplx val = it.value();
                for (int i = 0; i < N; ++i)
                {
                    // Row-major vec: idx = i*N + j
                    // C*A -> (I ⊗ A^T) vec(C): row = i*N + a_col, col = i*N + a_row
                    const int row = i * N + a_col;
                    const int col = i * N + a_row;
                    triplets.emplace_back(row, col, val);
                }
            }
        }

        SpMat K(N * N, N * N);
        K.setFromTriplets(triplets.begin(), triplets.end());
        return K;
    }

    // ---------- Projectors ----------
    inline SpMat projector_centered_sp(int L, int size, const std::string &bc = "open")
    {
        const int N = 2 * size;
        const int center = size;
        const int radius = std::abs(L);
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(N);

        if (bc == "open")
        {
            const int lo = std::max(0, center - radius);
            const int hi = std::min(N - 1, center + radius);
            for (int i = lo; i <= hi; ++i)
                diag(i) = 1.0;
        }
        else if (bc == "periodic" || bc == "pbc")
        {
            for (int i = center - radius; i <= center + radius; ++i)
                diag((i % N + N) % N) = 1.0;
        }
        else
        {
            throw std::runtime_error("bc must be 'open' or 'periodic'");
        }

        SpMat P(N, N);
        P.setIdentity();
        for (int i = 0; i < N; ++i)
            P.coeffRef(i, i) = diag(i);
        return P;
    }

    inline SpMat projector_prefix_sp(int LA, int N, const std::string &bc = "open")
    {
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(N);
        if (LA > 0)
        {
            const int limit = std::min(LA, N);
            for (int i = 0; i < limit; ++i)
                diag(i) = 1.0;
        }
        SpMat P(N, N);
        P.setIdentity();
        for (int i = 0; i < N; ++i)
            P.coeffRef(i, i) = diag(i);
        return P;
    }

    inline SpMat projector_prefix_RIGHT_sp(int LA, int N, const std::string &bc = "open")
    {
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(N);
        if (LA > 0)
        {
            const int start = std::max(0, LA);
            for (int i = start; i < N; ++i)
                diag(i) = 1.0;
        }
        SpMat P(N, N);
        P.setIdentity();
        for (int i = 0; i < N; ++i)
            P.coeffRef(i, i) = diag(i);
        return P;
    }

    // ---------- Liouvillian ----------
    inline SpMat liouvillian_sp(const SpMat &L, const SpMat &P, double gamma)
    {
        SpMat L_sup = spre_sp(L) + spost_sp(L.adjoint());
        SpMat proj = spre_sp(P) * spost_sp(P);
        L_sup += proj * (2.0 * gamma);
        return L_sup;
    }

    // ---------- Krylov-based expm_multiply (Al-Mohy/Higham style) ----------
    // Approximates exp(t*A)*v without forming exp(t*A):
    // 1) Build an Arnoldi basis V (size up to m) for span{v, A v, ...}.
    // 2) Project: H = V^* (t*A) V (small Hessenberg).
    // 3) Compute f = exp(H) (beta * e1) exactly (dense small matrix).
    // 4) Lift back: result = V * f. Early exit if subdiagonal drops below tol.
    inline Vec expm_multiply_krylov(const SpMat &A, const Vec &v, double t, int m = 30, double tol = 1e-10)
    {
        const double beta = v.norm();
        if (beta == 0.0)
            return Vec::Zero(v.size());

        std::vector<Vec> V;
        V.reserve(m + 1);
        V.push_back(v / beta); // V[0] is the first basis vector

        // H is (m+1) x m to hold the subdiagonal h_{j+1,j}; the usable square
        // block is H.topLeftCorner(k_use, k_use)
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(m + 1, m);
        int k_use = m;

        for (int j = 0; j < m; ++j)
        {
            Vec w = t * (A * V[j]);
            for (int i = 0; i <= j; ++i)
            {
                H(i, j) = V[i].dot(w);
                w -= H(i, j) * V[i];
            }
            const double h_next = w.norm();
            H(j + 1, j) = h_next; // subdiagonal element
            if (h_next < tol)
            {
                k_use = j + 1; // early convergence: use (j+1) basis vectors
                break;
            }
            V.push_back(w / h_next);
        }

        Eigen::MatrixXcd Hm = H.topLeftCorner(k_use, k_use);
        Eigen::MatrixXcd expHm = Hm.exp();

        Eigen::VectorXcd e1 = Eigen::VectorXcd::Zero(k_use);
        e1(0) = beta;
        Eigen::VectorXcd f = expHm * e1; // coefficients in Krylov basis

        Vec result = Vec::Zero(v.size());
        for (int i = 0; i < k_use; ++i)
        {
            result += f(i) * V[i];
        }
        return result;
    }

    // Adaptive expm_multiply with step-doubling error control.
    inline Vec expm_multiply_adaptive(const SpMat &A, const Vec &v, double t,
                                      int m = 30, double tol = 1e-10, double dt_max = 1.0)
    {
        if (t == 0.0)
            return v;
        const double t_total = t;
        Vec y = v;
        double t_done = 0.0;
        double dt = std::min(dt_max, std::abs(t_total));
        const double sign = (t_total >= 0.0) ? 1.0 : -1.0;

        while (t_done < std::abs(t_total))
        {
            if (t_done + dt > std::abs(t_total))
                dt = std::abs(t_total) - t_done;

            // One step
            Vec y1 = expm_multiply_krylov(A, y, sign * dt, m, tol);
            // Two half steps
            Vec y_half = expm_multiply_krylov(A, y, sign * (dt * 0.5), m, tol);
            Vec y2 = expm_multiply_krylov(A, y_half, sign * (dt * 0.5), m, tol);

            double err = (y2 - y1).norm();
            double denom = std::max(1.0, y2.norm());
            double rel_err = err / denom;

            if (rel_err > 5.0 * tol && dt > 1e-12)
            {
                dt *= 0.5; // reduce step and retry
                continue;
            }

            // accept the more accurate y2
            y = y2;
            t_done += dt;

            // increase step if error very small
            if (rel_err < 0.1 * tol)
                dt = std::min(dt * 1.5, dt_max);
        }
        return y;
    }

    // Batch evolution for multiple times (independent calls).
    inline std::vector<Vec> evolve_vec_times(const SpMat &L_sup,
                                             const Vec &vecC0,
                                             const std::vector<double> &t_array,
                                             int m = 30,
                                             double tol = 1e-10,
                                             double dt_max = 1.0)
    {
        std::vector<Vec> out;
        out.reserve(t_array.size());
        for (double t : t_array)
        {
            out.push_back(expm_multiply_adaptive(L_sup, vecC0, t, m, tol, dt_max));
        }
        return out;
    }

    // Single-time convenience wrapper (adaptive)
    inline Vec evolve_vec_single(const SpMat &L_sup,
                                 const Vec &vecC0,
                                 double t,
                                 int m = 30,
                                 double tol = 1e-10,
                                 double dt_max = 1.0)
    {
        return expm_multiply_adaptive(L_sup, vecC0, t, m, tol, dt_max);
    }

} // namespace dyn
