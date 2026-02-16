// ghd_core.hpp

#pragma once

#include <vector>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ghd
{

    constexpr double PI = 3.14159265358979323846;

    // ============ 1. Truncated system (direct solve) ===============//
    // Solve for b_r (r = 1..M) from:
    // b_r = base_r + (1/(gamma * r * PI * J)) sum_n b_n f_{n,r}
    // with
    //   base_r = 1/(2*pi^2*r) (1 - (-1)^r)
    //   f_{n,r} = (32 J^2 r^2)/(1 - 4 r^2)             if n == r
    //           = (16 J^2 n r (1 + (-1)^{n+r})) /
    //             (n^4 - 2 n^2 (r^2+1) + (r^2 - 1)^2)   otherwise
    // Rearranged to linear system:
    //   (gamma r pi J) b_r - sum_n f_{n,r} b_n = (gamma r pi J) base_r
    // so A_{r,n} = (gamma r pi J) delta_{rn} - f_{n,r}.
    inline void SolveTruncatedSystemDirect(int M, double J, double gamma,
                                           std::vector<int> &rVals,
                                           Eigen::VectorXd &bVals)
    {
        rVals.resize(M);
        for (int i = 0; i < M; ++i)
            rVals[i] = i + 1; // r = 1..M

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(M, M);
        Eigen::VectorXd rhs(M);

        auto base_term = [](int r) -> double
        {
            return (1.0 / (2.0 * PI * PI * r)) * (1.0 - std::pow(-1.0, r));
        };

        for (int rIdx = 0; rIdx < M; ++rIdx)
        {
            const int r = rVals[rIdx];
            for (int nIdx = 0; nIdx < M; ++nIdx)
            {
                const int n = rVals[nIdx];
                double f = 0.0;
                if (n == r)
                {
                    f = (32.0 * J * J * r * r) / (1.0 - 4.0 * r * r);
                }
                else
                {
                    const double parity = 1.0 + std::pow(-1.0, n + r);
                    const double denom = std::pow(n, 4) - 2.0 * n * n * (r * r + 1.0) + std::pow(r * r - 1.0, 2);
                    f = (16.0 * J * J * n * r * parity) / denom;
                }
                double diag = (nIdx == rIdx) ? (gamma * r * PI * J) : 0.0;
                A(rIdx, nIdx) = diag - f;
            }
            rhs(rIdx) = (gamma * r * PI * J) * base_term(r);
        }

        bVals = A.fullPivLu().solve(rhs);
    }

    // ========= 2. \Chi(k), \Theta, n_L, n_R, n_\zeta(k)=========

    inline double Theta(double x)
    {
        if (x > 0.0)
            return 1.0;
        if (x < 0.0)
            return 0.0;
        return 0.5;
    }

    inline double nL(double /*k*/) { return 0.0; }
    inline double nR(double /*k*/) { return 1.0 / (4.0 * PI); }

    // \chi^+(k)= Theta(k) * chi^R(k) + Theta(-k) * chi^L(k)
    // chi^R(k) = sum_r b_r sin(r k)
    // chi^L(k) = 1/(4 pi) + sum_r b_r sin(r k)
    inline double ChiPlus(double k,
                          const std::vector<int> &rVals,
                          const Eigen::VectorXd &bVals)
    {
        double s = 0.0;
        const int dim = static_cast<int>(rVals.size());
        for (int i = 0; i < dim; ++i)
        {
            s += bVals(i) * std::sin(rVals[i] * k);
        }
        if (k > 0.0)
        {
            return s;
        }
        else
        {
            return (1.0 / (4.0 * PI)) + s;
        }
    }

    // n_\zeta(k) with \epsilon'(k)= 2 J\sin k (J=1)

    inline double nZeta(double k, double zeta, double J,
                        const std::vector<int> &rVals,
                        const Eigen::VectorXd &bVals)
    {
        const double epsp = 2.0 * J * std::sin(k);
        const double chi = ChiPlus(k, rVals, bVals);

        if (k > 0.0)
        {
            return Theta(-zeta) * nL(k) + Theta(zeta) * Theta(epsp - zeta) * chi + Theta(zeta - epsp) * nR(k);
        }
        else
        {
            return Theta(zeta) * nR(k) + Theta(-zeta) * Theta(-epsp + zeta) * chi + Theta(epsp - zeta) * nL(k);
        }
    }

    inline double qPlus(double k, int r, double J) { return 2.0 * J * std::cos(r * k); }
    inline double qMinus(double k, int r, double J) { return -2.0 * J * std::sin(r * k); }

    // ============= 3. HydCharge / HydCurrent===========
    //
    // We parametrise the integrator as:
    //   double integrate(const std::function<double(double)>& f,
    //                    double a, double b);
    //
    // so you can plug in Gauss-Legendre or QAGS.
    // =================================================

    using IntegratorFn = std::function<double(const std::function<double(double)> &, double, double)>;

    // q_r(\zeta)
    inline double HydCharge(int r, double J, double zeta, char sign,
                            const std::vector<int> &rVals,
                            const Eigen::VectorXd &bVals,
                            const IntegratorFn &integrate)
    {
        std::function<double(double)> f =
            [=, &rVals, &bVals](double k) -> double
        {
            double q = (sign == '-') ? qMinus(k, r, J) : qPlus(k, r, J);
            return q * nZeta(k, zeta, J, rVals, bVals);
        };
        return integrate(f, -PI, PI);
    }

    // J_r(\zeta)
    inline double HydCurrent(int r, double J, double zeta, char sign, const std::vector<int> &rVals,
                             const Eigen::VectorXd &bVals,
                             const IntegratorFn &integrate)
    {
        std::function<double(double)> f =
            [=, &rVals, &bVals](double k) -> double
        {
            double q = (sign == '-') ? qMinus(k, r, J) : qPlus(k, r, J);
            return 2.0 * J * std::sin(k) * q * nZeta(k, zeta, J, rVals, bVals);
        };
        return integrate(f, -PI, PI);
    }

    // ============= 4. Profile computation & CSV =============

    struct ProfileResult
    {

        std::vector<double> zetas;
        std::vector<double> qVals;
        std::vector<double> JVals;
        std::vector<int> rVals;
        Eigen::VectorXd bVals;
    };

    inline ProfileResult ComputeGHDProfiles(
        int M, double J, double gamma,
        int rCharge, char sign,
        double zetaMin, double zetaMax, int numZeta,
        const IntegratorFn &integrate,
        bool useOpenMP = true)
    {
        using clock = std::chrono::steady_clock;

        ProfileResult res;

        // --1) Solve truncated system ---
        auto t0 = clock::now();
        SolveTruncatedSystemDirect(M, J, gamma, res.rVals, res.bVals);
        auto t1 = clock::now();
        double tSolve_seconds = std::chrono::duration<double>(t1 - t0).count();
        double tSolve_minutes = tSolve_seconds / 60.0;
        std::cout << "SolveTruncateSystemDirect took "
                  << tSolve_minutes << " min\n";

        // --- 2) Build zeta grid
        res.zetas.resize(numZeta);
        const double dz = (zetaMax - zetaMin) / (numZeta - 1);
        for (int i = 0; i < numZeta; ++i)
        {
            res.zetas[i] = zetaMin + i * dz;
        }
        res.qVals.resize(numZeta);
        res.JVals.resize(numZeta);

        // --3) Hydrodynamic integrals over \zeta
        auto t2 = clock::now();

        // sign "+" or "-"
        char sgn = sign;

// COMPUTATION IN PARALLEL
#pragma omp parallel for if (useOpenMP) schedule(dynamic)
        for (int i = 0; i < numZeta; ++i)
        {
            double z = res.zetas[i];
            res.qVals[i] = HydCharge(rCharge, J, z, sgn, res.rVals, res.bVals, integrate);

            res.JVals[i] = HydCurrent(rCharge, J, z, sgn, res.rVals, res.bVals, integrate);
        }

        auto t3 = clock::now();
        double tIntegrate_seconds = std::chrono::duration<double>(t3 - t2).count();
        double tIntegrate_minutes = tIntegrate_seconds / 60.0;
        std::cout << "GHD integrals (all rays) took "
                  << tIntegrate_minutes << " min\n";

        return res;
    }

    // Write CSV: columns--> \zeta, q(\zeta), J(\zeta)
    inline void write_csv(const std::string &filename,
                          const ProfileResult &res)
    {
        std::ofstream out(filename);
        if (!out)
        {
            std::cerr << "Could not open" << filename << " for writing\n";
            return;
        }
        out << "zeta,q,J\n";
        for (std::size_t i = 0; i < res.zetas.size(); ++i)
        {
            out << res.zetas[i] << ","
                << res.qVals[i] << ","
                << res.JVals[i] << "\n";
        }
        out.close();
        std::cout << "Wrote CSV:" << filename << "\n";
    }

} // namespace ghd
