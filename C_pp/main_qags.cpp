// main_qags.cpp
#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/ghd_core.hpp"
#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/qags_gsl.hpp"

using namespace ghd;
/*
Use the following command to get the binary file

 g++ -std=c++17 -O2 main_qags.cpp -I/opt/homebrew/include/eigen3 -I/opt/homebrew/include -L/opt/homebrew/lib -lgsl -lgslcblas -o main_qags

 O2 turns on compiler optimizations !

 Use the following command to get the output
 ./main_qags

*/

int main(int argc, char **argv)
{
    int M = 35;
    double Jval = 1.0;
    double gamma = 1.0;
    int rcharge = 0;
    char sign = '+';
    double zMin = -3.0;
    double zMax = 3.0;
    int nZ = 300;

    if (argc > 1)
    {
        gamma = std::stod(argv[1]);
    }
    if (argc > 2)
    {
        M = std::stoi(argv[2]);
    }
    if (argc > 3)
    {
        rcharge = std::stoi(argv[3]);
    }
    if (argc > 4 && argv[4])
    {
        sign = argv[4][0];
    }
    double epsabs = 1e-13;
    double epsrel = 1e-13;
    std::size_t limit = 5000;
    int min_rec = 0; // number of forced bisections (Mathematica MinRecursion)
    int max_rec = -1; // optional cap on recursion depth -> translated into workspace limit
    if (argc > 5)
    {
        epsabs = std::stod(argv[5]);
    }
    if (argc > 6)
    {
        epsrel = std::stod(argv[6]);
    }
    if (argc > 7)
    {
        limit = static_cast<std::size_t>(std::stoul(argv[7]));
    }
    if (argc > 8)
    {
        min_rec = std::stoi(argv[8]);
    }
    if (argc > 9)
    {
        max_rec = std::stoi(argv[9]);
    }

    // Translate max_recursion to a workspace limit proxy: limit >= 2^(max_recursion+1)-1
    if (max_rec >= 0 && max_rec < 30)
    {
        std::size_t needed = (1u << (max_rec + 1)) - 1u;
        if (needed > limit)
            limit = needed;
    }

    // tighter tolerances and larger workspace to mirror slower, high-accuracy runs
    IntegratorFn integrator = make_qags_integrator(epsabs, epsrel, limit, min_rec);

    std::cout << "Running GHD with QAGS (GSL) integrator...\n";

    auto t0 = std::chrono::steady_clock::now();
    ProfileResult res = ComputeGHDProfiles(
        M, Jval, gamma,
        rcharge, sign,
        zMin, zMax, nZ,
        integrator,
        /*useOpenMP=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double tTotal = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Total run time (QAGS) = " << tTotal << " s\n";

    write_csv("GHD_r" + std::to_string(rcharge) +
                  "_sign" + std::string(1, sign) +
                  "_M" + std::to_string(M) +
                  "_gamma" + std::to_string(gamma) +
                  "_qags.csv",
              res);

    return 0;
}
