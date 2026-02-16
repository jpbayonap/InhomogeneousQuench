// main_dynamics_BDY.cpp
// Boundary test for r=1, sign = "-", varying gamma values.
// Loops over sizes/times arrays, evaluates conditions 3 and 4, and writes a CSV
// with gamma, zeta0, cond3 (complex), cond4 (complex), time, and size.

#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/dynamics_module.hpp"
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using cplx = std::complex<double>;
using Mat = dyn::Mat;

// Safe element access with boundary conditions.
inline cplx get_elem(const Mat &C, int i, int j, const std::string &bc)
{
    const int N = static_cast<int>(C.rows());
    if (!bc.empty() && (bc[0] == 'p' || bc[0] == 'P'))
    {
        // periodic: wrap indices
        int ii = ((i % N) + N) % N;
        int jj = ((j % N) + N) % N;
        return C(ii, jj);
    }
    // open: zero outside bounds
    if (i >= 0 && i < N && j >= 0 && j < N)
        return C(i, j);
    return cplx{0.0, 0.0};
}

// Symmetric local charges/currents (row-stacked convention preserved in mat2vec/vec2mat).
inline cplx qp_symm(int r, int x, const Mat &C, const std::string &bc)
{
    return get_elem(C, x - r, x + r, bc) + get_elem(C, x + r, x - r, bc);
}

inline cplx qm_symm(int r, int x, const Mat &C, const std::string &bc)
{
    return cplx{0.0, 1.0} * (get_elem(C, x - r, x + r + 1, bc) - get_elem(C, x + r + 1, x - r, bc));
}

inline cplx jm_symm(int r, int x, const Mat &C, const std::string &bc)
{
    return -(get_elem(C, x - r + 1, x + r + 1, bc) - get_elem(C, x - r, x + r + 2, bc) - get_elem(C, x + r + 2, x - r, bc) + get_elem(C, x + r + 1, x - r + 1, bc));
}

int main()
{
    using namespace dyn;

    // Parameters (small defaults to test; adjust sizes/target_zeta for large runs)
    std::vector<int> sizes{400};                                               // half-chain L; set to {800, 1000, ...} for production
    std::vector<double> target_zeta{1000, 5000, 10000, 50000, 100000, 500000}; // treated here as times to probe smaller zeta0
    std::vector<double> gammas{0, 0.001, 0.01, 0.1, 0.5, 1.0};
    int r = 0;
    int OFFSET = 1;
    double J1 = 1.0;
    double mu = 0.0;
    std::string bc = "open"; // map Python "obc" to "open"

    // Output file: encode size range to avoid confusion when multiple sizes are run
    int minL = *std::min_element(sizes.begin(), sizes.end());
    int maxL = *std::max_element(sizes.begin(), sizes.end());
    std::string outname;
    if (minL == maxL)
        outname = "GHD_BDY_r1_sign-_N" + std::to_string(2 * minL) + ".csv";
    else
        outname = "GHD_BDY_r1_sign-_N" + std::to_string(2 * minL) + "_to_" + std::to_string(2 * maxL) + ".csv";
    std::ofstream out(outname);
    if (!out)
    {
        std::cerr << "Could not open " << outname << " for writing\n";
        return 1;
    }
    out << "gamma,zeta0,cond3,cond3_abs,cond4,cond4_abs,time,size\n";
    out << std::setprecision(15);

    for (int s : sizes)
    {
        int N = 2 * s;
        int center = N / 2;
        int i_plus = center + OFFSET;
        int i_minus = center - OFFSET;

        // Build Hamiltonian and projector
        dyn::SpMat H = Hamiltonian(s, J1, mu, bc);
        dyn::SpMat P_right = projector_prefix_RIGHT_sp(/*LA=*/s, /*N=*/N, bc);

        // Initial state
        Mat C0 = dyn::Gamma0(s);
        dyn::Vec vecC0 = dyn::mat2vec(C0);

        for (double T : target_zeta)
        {
            double zeta0 = static_cast<double>(i_plus) / T; // ~offset / T near interface

            for (double g : gammas)
            {
                // h_cond = i*H - g * P_right
                dyn::SpMat h_cond = (cplx{0.0, 1.0}) * H - g * P_right;
                dyn::SpMat L_sup = liouvillian_sp(h_cond, P_right, g);

                dyn::Vec vecC_t = evolve_vec_single(L_sup, vecC0, T);
                Mat C_zeta = dyn::vec2mat(vecC_t, N);

                // Condition 3: j_delta + (2r+1) * gamma * q0
                cplx j_delta = jm_symm(r, i_plus, C_zeta, bc) - jm_symm(r, i_minus, C_zeta, bc);
                cplx q0 = qm_symm(r, center, C_zeta, bc);
                cplx cond3_c = j_delta + (2 * r + 1) * g * q0;
                double cond3 = cond3_c.real();
                double cond3_abs = std::abs(cond3_c);

                // Condition 4 (only r=1): j_delta_m + J1^2
                cplx j_delta_m = jm_symm(r, i_plus, C_zeta, bc) + jm_symm(r, i_minus, C_zeta, bc);
                cplx cond4_c = j_delta_m;
                double cond4 = cond4_c.real();
                double cond4_abs = std::abs(cond4_c);

                out << g << "," << zeta0 << ","
                    << cond3 << "," << cond3_abs << ","
                    << cond4 << "," << cond4_abs << ","
                    << T << "," << N << "\n";
            }
            std::cout << "done with time evolution, size=" << N << ", T=" << T << ", offset=" << OFFSET << "\n";
        }
    }

    std::cout << "Wrote CSV: " << outname << "\n";
    return 0;
}
