// main_gauss.cpp
#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/ghd_core.hpp"
#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/gauss_legendre.hpp"

using namespace ghd;

/*
Use the following command to get the binary file

g++ -std=c++17 -O2 main_gauss.cpp -I/opt/homebrew/include/eigen3 -o main_gauss

O2 turns on compiler optimizations !

Use the following command to get the output
./main_gauss

*/

int main(int argc, char **argv)
{
    // Parameters
    int M = 200; // truncation in r sum
    double Jval = 1.0;
    double gamma = 1.0;
    int rcharge = 2;
    char sign = '+'; // '+' or '-'
    double zMin = -3.0;
    double zMax = 3.0;
    int nZ = 300;
    int gauss_order = 128;
    int min_rec = 0;
    std::size_t max_splits = 0;

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
    if (argc > 5)
    {
        gauss_order = std::stoi(argv[5]);
    }
    if (argc > 6)
    {
        min_rec = std::stoi(argv[6]);
    }
    if (argc > 7)
    {
        max_splits = static_cast<std::size_t>(std::stoul(argv[7]));
    }

    // Choose Gauss-Legendre integrator with optional forced splits
    IntegratorFn integrator = make_gauss_integrator(gauss_order, min_rec, max_splits);

    std::cout << "Running GHD with Gauss-Legendre integrator ...\n";

    auto t0 = std::chrono::steady_clock::now();
    ProfileResult res = ComputeGHDProfiles(
        M, Jval, gamma, rcharge, sign,
        zMin, zMax, nZ,
        integrator,
        /* useOpenMP*/ false);
    auto t1 = std::chrono::steady_clock::now();
    double tTotal_seconds = std::chrono::duration<double>(t1 - t0).count();
    double tTotal_Minutes = tTotal_seconds / 60.0;
    std::cout << "Total run time (Gauss) = " << tTotal_Minutes << " min\n";

    // Write CSV
    write_csv("GHD_r" + std::to_string(rcharge) +
                  "_sign" + std::string(1, sign) +
                  "_M" + std::to_string(M) +
                  "_gamma" + std::to_string(gamma) +
                  "_gauss.csv",
              res);

    return 0;
}
