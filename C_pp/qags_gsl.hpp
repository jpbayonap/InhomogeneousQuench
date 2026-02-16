// qags_gsl.hpp
#pragma once
#include "/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/ghd_core.hpp"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <iostream>
#include <algorithm>

namespace ghd
{

    struct GSLFunctorData
    {
        const std::function<double(double)> *f;
    };

    inline double gsl_wrapper(double x, void *params)
    {
        auto *data = static_cast<GSLFunctorData *>(params);
        return (*(data->f))(x);
    }

    inline double integrate_qags(const std::function<double(double)> &f,
                                 double a, double b,
                                 double epsabs = 1e-9, double epsrel = 1e-9,
                                 std::size_t limit = 1000,
                                 int min_recursion = 0)
    {
        // prevent GSL from aborting on warnings (roundoff, etc.)
        gsl_set_error_handler_off();

        // If min_recursion>0, pre-split into 2^min_recursion subintervals like NIntegrate
        const int splits = std::max(1, 1 << std::max(0, min_recursion));
        const double h = (b - a) / static_cast<double>(splits);

        auto integrate_segment = [&](double aa, double bb, double epsabs_seg, double epsrel_seg, std::size_t lim_seg)
        {
            gsl_integration_workspace *w = gsl_integration_workspace_alloc(lim_seg);
            GSLFunctorData data{&f};
            gsl_function F;
            F.function = &gsl_wrapper;
            F.params = &data;

            double result = 0.0, error = 0.0;
            int status = gsl_integration_qags(&F, aa, bb, epsabs_seg, epsrel_seg, lim_seg, w, &result, &error);
            if (status != GSL_SUCCESS)
            {
                // retry once with looser tolerances and larger workspace
                gsl_integration_workspace_free(w);
                std::size_t limit2 = std::max<std::size_t>(2000, lim_seg * 2);
                w = gsl_integration_workspace_alloc(limit2);
                status = gsl_integration_qags(&F, aa, bb, epsabs_seg * 10.0, epsrel_seg * 10.0, limit2, w, &result, &error);
                if (status != GSL_SUCCESS)
                {
                    std::cerr << "GSL QAGS warning: status=" << status
                              << " err~" << error << " on [" << aa << "," << bb << "]\n";
                }
            }
            gsl_integration_workspace_free(w);
            return result;
        };

        double total = 0.0;
        // distribute abs tol across splits; keep rel tol unchanged
        double epsabs_seg = epsabs / static_cast<double>(splits);
        for (int i = 0; i < splits; ++i)
        {
            double aa = a + i * h;
            double bb = aa + h;
            total += integrate_segment(aa, bb, epsabs_seg, epsrel, limit);
        }
        return total;
    }

    inline IntegratorFn make_qags_integrator(double epsabs = 1e-9,
                                             double epsrel = 1e-9,
                                             std::size_t limit = 1000,
                                             int min_recursion = 0)
    {
        return [=](const std::function<double(double)> &f,
                   double a, double b)
        {
            return integrate_qags(f, a, b, epsabs, epsrel, limit, min_recursion);
        };
    }

} // namespace ghd
