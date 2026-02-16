// main_dynamics_thermal.cpp
// Matrix-free RK evolution of thermal inhomogeneous initial state (betaL, betaR)
// Saves q/J profiles for given r, sign, gammas, size.

#include "dynamics_module.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

using namespace dyn;
using std::string;
using std::vector;
using cplx = std::complex<double>;
using RowMat = Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Safe element access
inline cplx get_elem(const Mat &C, int i, int j, const std::string &bc)
{
    const int N = static_cast<int>(C.rows());
    if (!bc.empty() && (bc[0] == 'p' || bc[0] == 'P'))
    {
        int ii = ((i % N) + N) % N;
        int jj = ((j % N) + N) % N;
        return C(ii, jj);
    }
    if (i >= 0 && i < N && j >= 0 && j < N)
        return C(i, j);
    return cplx{0.0, 0.0};
}

// Symmetric local charges/currents supporting even/odd r
inline cplx qp_symm(int r, int x, const Mat &C, const std::string &bc)
{
    if (r % 2 != 0)
    {
        int R = (r - 1) / 2;
        return get_elem(C, x - R, x + R + 1, bc) + get_elem(C, x + R + 1, x - R, bc);
    }
    int R = r / 2;
    return get_elem(C, x - R, x + R, bc) + get_elem(C, x + R, x - R, bc);
}

inline cplx qm_symm(int r, int x, const Mat &C, const std::string &bc)
{
    if (r % 2 != 0)
    {
        int R = (r - 1) / 2;
        return cplx{0.0, 1.0} * (get_elem(C, x - R, x + R + 1, bc) - get_elem(C, x + R + 1, x - R, bc));
    }
    int R = r / 2;
    return cplx{0.0, 1.0} * (get_elem(C, x - R, x + R, bc) - get_elem(C, x + R, x - R, bc));
}

inline cplx jp_symm(int r, int x, const Mat &C, const std::string &bc)
{
    if (r % 2 != 0)
    {
        int R = (r - 1) / 2;
        return cplx{0.0, 1.0} *
               (get_elem(C, x - R + 1, x + R + 1, bc) - get_elem(C, x - R, x + R + 2, bc) +
                get_elem(C, x + R + 2, x - R, bc) - get_elem(C, x + R + 1, x - R + 1, bc));
    }
    int R = r / 2;
    return cplx{0.0, 1.0} *
           (get_elem(C, x - R + 1, x + R, bc) - get_elem(C, x - R, x + R + 1, bc) +
            get_elem(C, x + R + 1, x - R, bc) - get_elem(C, x + R, x - R + 1, bc));
}

inline cplx jm_symm(int r, int x, const Mat &C, const std::string &bc)
{
    if (r % 2 != 0)
    {
        int R = (r - 1) / 2;
        return -(get_elem(C, x - R + 1, x + R + 1, bc) - get_elem(C, x - R, x + R + 2, bc) -
                 get_elem(C, x + R + 2, x - R, bc) + get_elem(C, x + R + 1, x - R + 1, bc));
    }
    int R = r / 2;
    return -(get_elem(C, x - R + 1, x + R, bc) - get_elem(C, x - R, x + R + 1, bc) -
             get_elem(C, x + R + 1, x - R, bc) + get_elem(C, x + R, x - R + 1, bc));
}

// Thermal block correlations via Fermi factors f(eps)=1/(1+exp(beta*eps))
inline Mat Gamma_beta(double beta, const SpMat &H_block)
{
    Eigen::MatrixXd Hd = Eigen::MatrixXd(H_block.real());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXd evecs = es.eigenvectors();
    Eigen::VectorXd f_occ = 1.0 / (1.0 + (beta * evals.array()).exp());
    Eigen::MatrixXd C = evecs * f_occ.asDiagonal() * evecs.transpose();
    return C.cast<cplx>();
}

int main(int argc, char **argv)
{
    // Args: L r sign betaL betaR T rk_steps gamma1 gamma2 ...
    int L = 600;
    int r = 1;
    char sign = '+';
    double betaL = 1.0;
    double betaR = 1.0;
    double T = 200.0;
    int rk_steps = 400;
    std::string outdir = ".";
    vector<double> gammas{0.0};

    if (argc > 1)
        L = std::stoi(argv[1]);
    if (argc > 2)
        r = std::stoi(argv[2]);
    if (argc > 3 && argv[3])
        sign = argv[3][0];
    if (argc > 4)
        betaL = std::stod(argv[4]);
    if (argc > 5)
        betaR = std::stod(argv[5]);
    if (argc > 6)
        T = std::stod(argv[6]);
    if (argc > 7)
        outdir = argv[7];
    if (argc > 8)
        rk_steps = std::stoi(argv[8]);
    if (argc > 9)
    {
        gammas.clear();
        for (int i = 9; i < argc; ++i)
            gammas.push_back(std::stod(argv[i]));
    }

    const int N = 2 * L;
    const int center = N / 2;
    std::string bc = "open";

    // Build H and P
    SpMat H = Hamiltonian(L, 1.0, 0.0, bc);
    SpMat P_right = projector_prefix_RIGHT_sp(L, N, bc);

    // Thermal initial state
    // Extract blocks
    SpMat HR = H.bottomRightCorner(L, L);
    SpMat HL = H.topLeftCorner(L, L);
    Mat C_R = Gamma_beta(betaR, HR);
    Mat C0 = Mat::Zero(N, N);
    if (betaL != 0.0)
    {
        Mat C_L = Gamma_beta(betaL, HL);
        C0.topLeftCorner(L, L) = C_L;
    }
    C0.bottomRightCorner(L, L) = C_R;
    Vec vec0 = mat2vec(C0);
    // Directory for saving data and pictures
    std::filesystem::path csv_dir = std::filesystem::path(outdir) / "GHD_THERM_CSV_CPP";
    std::error_code ec;
    std::filesystem::create_directories(csv_dir, ec);

    for (double g : gammas)
    {
        std::ostringstream oss;
        oss << "GHD_THERM_cpp_r" << r << "_sign" << sign << "_gamma" << std::fixed << std::setprecision(2) << g << "_N" << N << ".csv";
        std::string outcsv = (csv_dir / oss.str()).string();
        // open the outcsv file to insert data
        std::ofstream out(outcsv);
        out << "gamma,time,zeta,q,j\n";
        out << std::setprecision(15);

        // matrix-free matvec
        auto matvec = [&](const Vec &v) -> Vec
        {
            RowMat C = Eigen::Map<const RowMat>(v.data(), N, N);
            RowMat term1 = H * C;
            RowMat term2 = C * H.adjoint();
            RowMat term3 = 2.0 * g * (P_right * C * P_right.transpose());
            RowMat res = term1 + term2 + term3;
            return Eigen::Map<const Vec>(res.data(), res.size());
        };

        // RK4 evolve
        Vec v = vec0;
        double dt = T / static_cast<double>(rk_steps);
        for (int step = 0; step < rk_steps; ++step)
        {
            Vec k1 = matvec(v);
            Vec k2 = matvec(v + 0.5 * dt * k1);
            Vec k3 = matvec(v + 0.5 * dt * k2);
            Vec k4 = matvec(v + dt * k3);
            v = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        Mat C_t = vec2mat(v, N);

        for (int x = 0; x < N; ++x)
        {
            double zeta = static_cast<double>(x - center) / T;
            double q_val = 0.0, j_val = 0.0;
            if (sign == '+')
            {
                q_val = std::real(qp_symm(r, x, C_t, bc));
                j_val = std::real(jp_symm(r, x, C_t, bc));
            }
            else
            {
                q_val = std::real(qm_symm(r, x, C_t, bc));
                j_val = std::real(jm_symm(r, x, C_t, bc));
            }
            out << g << "," << T << "," << zeta << "," << q_val << "," << j_val << "\n";
        }
        std::cout << "Wrote " << outcsv << "\n";
    }

    return 0;
}
