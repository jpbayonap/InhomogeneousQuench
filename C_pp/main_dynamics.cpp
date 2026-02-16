// main_dynamics.cpp
// Compute q^-(r=1) and J^-(r=1) profiles for multiple sizes/times/gammas.
// Writes CSV with gamma, time, size, site index, zeta, q_minus, j_minus.

#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/dynamics_module.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <future>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <sstream> // to pass " " separated lists

using cplx = std::complex<double>;
using Mat = dyn::Mat;

// Safe element access with boundary conditions.
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
               (get_elem(C, x - R + 1, x + R + 1, bc) -
                get_elem(C, x - R, x + R + 2, bc) +
                get_elem(C, x + R + 2, x - R, bc) -
                get_elem(C, x + R + 1, x - R + 1, bc));
    }
    int R = r / 2;
    return cplx{0.0, 1.0} *
           (get_elem(C, x - R + 1, x + R, bc) -
            get_elem(C, x - R, x + R + 1, bc) +
            get_elem(C, x + R + 1, x - R, bc) -
            get_elem(C, x + R, x - R + 1, bc));
}

inline cplx jm_symm(int r, int x, const Mat &C, const std::string &bc)
{
    if (r % 2 != 0)
    {
        int R = (r - 1) / 2;
        return -(get_elem(C, x - R + 1, x + R + 1, bc) -
                 get_elem(C, x - R, x + R + 2, bc) -
                 get_elem(C, x + R + 2, x - R, bc) +
                 get_elem(C, x + R + 1, x - R + 1, bc));
    }
    int R = r / 2;
    return -(get_elem(C, x - R + 1, x + R, bc) -
             get_elem(C, x - R, x + R + 1, bc) -
             get_elem(C, x + R + 1, x - R, bc) +
             get_elem(C, x + R, x - R + 1, bc));
}

inline dyn::SpMat projector_RIGHT_len_sp(int LA, int L, int N)
{
    Eigen::VectorXd diag = Eigen::VectorXd::Zero(N);
    const int start = std::max(0, LA);
    const int end = std::min(N, LA + L + 1);
    for (int i = start; i < end; ++i)
        diag(i) = 1.0;
    dyn::SpMat P(N, N);
    P.setIdentity();
    for (int i = 0; i < N; ++i)
        P.coeffRef(i, i) = diag(i);
    return P;
}

// "1,2,3" -> vector<int>
static std::vector<int> parse_int_list(const std::string &s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
            out.push_back(std::stoi(item));
    }
    return out;
}

// "200.0, 400.0" -> vector<double>
static std::vector<double> parse_double_list(const std::string &s)
{
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
            out.push_back(std::stod(item));
    }
    return out;
}

int main(int argc, char **argv)
{
    using namespace dyn;

    struct Task
    {
        int s;
        double T;
        double g;
        int Llen;
    };

    // Parameters
    std::vector<int> sizes{500};          // half-chain L (can be overridden via argv)
    std::vector<double> target_zeta{200}; // times
    std::vector<double> gammas{0.0, 0.1, 0.5, 1.0};
    std::vector<int> lengths{1}; // monitored interval length L
    int r = 1;                   // lattice r index
    char sign = '-';             // '+' or '-'
    bool use_rk4 = false;
    bool rk_adapt = false;
    int rk_steps = 400;
    double rk_tol = 1e-6;
    std::string outdir = ".";

    if (argc > 1)
    {
        r = std::stoi(argv[1]);
    }
    if (argc > 2 && argv[2])
    {
        sign = argv[2][0];
    }
    if (argc > 3)
    {
        use_rk4 = std::stoi(argv[3]) != 0;
    }
    if (argc > 4)
    {
        rk_adapt = std::stoi(argv[4]) != 0;
    }
    if (argc > 5)
    {
        rk_steps = std::max(10, std::stoi(argv[5]));
    }
    if (argc > 6)
    {
        rk_tol = std::stod(argv[6]);
    }
    if (argc > 7 && argv[7])
    {
        sizes = parse_int_list(argv[7]);
    }
    if (argc > 8 && argv[8])
    {
        target_zeta = parse_double_list(argv[8]);
    }
    if (argc > 9 && argv[9])
    {
        gammas = parse_double_list(argv[9]);
    }
    if (argc > 10 && argv[10])
    {
        lengths = parse_int_list(argv[10]);
    }
    if (argc > 11 && argv[11])
    {
        outdir = argv[11];
    }

    double J1 = 1.0;
    double mu = 0.0;
    std::string bc = "open";

    std::filesystem::path csv_dir = std::filesystem::path(outdir) / "GHD_NEEL_PARTIAL_CSV_CPP";
    std::filesystem::create_directories(csv_dir);
    std::vector<Task> tasks;
    tasks.reserve(sizes.size() * target_zeta.size() * gammas.size() * lengths.size());
    for (int s : sizes)
        for (double T : target_zeta)
            for (double g : gammas)
                for (int Llen : lengths)
                    tasks.push_back({s, T, g, Llen});

    std::vector<std::future<void>> futures;

    for (const auto &task : tasks)
    {
        // What time is it?
        futures.push_back(std::async(std::launch::async, [&, task]()
                                     {
            int s = task.s;
            double T = task.T;
            double g = task.g;
            int Llen = task.Llen;
            int N = 2 * s;
            int center = N / 2;
            std::cout << "Starting future: g=" << g
                      << ", T=" << T
                      << ", N=" << N
                      << ", r=" << r
                      << ", sign=" << sign
                      << "\n";

            dyn::SpMat H = Hamiltonian(s, J1, mu, bc);
            dyn::SpMat P_right = projector_RIGHT_len_sp(/*LA=*/s, /*L=*/Llen, /*N=*/N);
            Mat C0 = dyn::Gamma0(s);
            dyn::Vec vecC0 = dyn::mat2vec(C0);

            dyn::SpMat h_cond = (cplx{0.0, 1.0}) * H - g * P_right;

            dyn::Vec vecC_t;
            if (use_rk4)
            {
                auto matvec = [&](const dyn::Vec &v) {
                    RowMat C = Eigen::Map<const RowMat>(v.data(), N, N);
                    RowMat res = h_cond*C + C* h_cond.adjoint()+ 2.0 * g * (P_right * C * P_right.transpose());

                    // Copy into a real Vec so the returned data stays valid after this lambda exits.
                    // Returning a Map to a local matrix would dangle and corrupt RK4.
                    dyn::Vec out(res.size());
                    Eigen::Map<RowMat>(out.data(), N, N) = res;
                    return out;

                };
                auto rk_step = [&](const dyn::Vec &v, double dt)
                {
                    dyn::Vec k1 = matvec(v);
                    dyn::Vec k2 = matvec(v + 0.5 * dt * k1);
                    dyn::Vec k3 = matvec(v + 0.5 * dt * k2);
                    dyn::Vec k4 = matvec(v + dt * k3);
                    return v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                };

                if (rk_adapt)
                {
                    double t = 0.0;
                    dyn::Vec v = vecC0;
                    double dt = T / static_cast<double>(std::max(20, rk_steps));
                    while (t < T)
                    {
                        if (t + dt > T)
                            dt = T - t;
                        dyn::Vec v_full = rk_step(v, dt);
                        dyn::Vec v_half = rk_step(v, 0.5 * dt);
                        v_half = rk_step(v_half, 0.5 * dt);
                        double err = (v_full - v_half).norm() / (v_half.norm() + 1e-14);
                        if (err < rk_tol || dt <= 1e-12)
                        {
                            v = v_half;
                            t += dt;
                            dt *= 1.5;
                        }
                        else
                        {
                            dt *= 0.5;
                        }
                    }
                    vecC_t = v;
                }
                else
                {
                    double dt = T / static_cast<double>(rk_steps);
                    dyn::Vec v = vecC0;
                    for (int step = 0; step < rk_steps; ++step)
                    {
                        v = rk_step(v, dt);
                    }
                    vecC_t = v;
                }
            }
            else
            {
                int krylov_dim = 60;
                double tol = 1e-12;
                double dt_max = 10.0;
                dyn::SpMat L_sup = liouvillian_sp(h_cond, P_right, g);
                vecC_t = evolve_vec_single(L_sup, vecC0, T, krylov_dim, tol, dt_max);
            }
            Mat C_zeta = dyn::vec2mat(vecC_t, N);

            std::vector<double> q_vals(static_cast<size_t>(N));
            std::vector<double> j_vals(static_cast<size_t>(N));
            std::vector<double> zetas(static_cast<size_t>(N));
            for (int x = 0; x < N; ++x)
            {
                zetas[x] = static_cast<double>(x - center) / T;
                if (sign == '-')
                {
                    q_vals[x] = std::real(qm_symm(r, x, C_zeta, bc));
                    j_vals[x] = std::real(jm_symm(r, x, C_zeta, bc));
                }
                else
                {
                    q_vals[x] = std::real(qp_symm(r, x, C_zeta, bc));
                    j_vals[x] = std::real(jp_symm(r, x, C_zeta, bc));
                }
            }

            

            std::ostringstream fname;
            fname << "GHD_VCNEEL_l" << Llen
                  << "_r" << r
                  << "_sign" << sign
                  << "_gamma" << std::fixed << std::setprecision(2) << g
                  << "_T" << T
                  << "_N" << N
                  << ".csv";
            std::filesystem::path outname = csv_dir / fname.str();
            std::ofstream out(outname.string());
            if (!out)
            {
                std::cerr << "Could not open " << outname.string() << " for writing\n";
                return;
            }
            out << "gamma,time,zeta,q,j,bdy_residual,bdy_log10_abs\n";
            out << std::setprecision(15);
            for (int x = 0; x < N; ++x)
            {
                out << g << "," << T << "," << zetas[x] << ","
                    << q_vals[x] << "," << j_vals[x] << "\n";
            }
            std::cout << "saved: " << outname.string() << "\n"; }));
    }

    for (auto &fut : futures)
        fut.get();

    return 0;
}
