// Benchmarks a heat-equation RK4 pipeline.

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>

#include "common/adapters.hpp"

namespace {

constexpr int kSteps = 1000;

template<class Lib, std::size_t N>
void BenchmarkHeatRk4(benchmark::State& state)
{
    using T   = typename Lib::scalar_t;
    using Mat = typename Lib::template matrix_t<N, N>;
    using Vec = typename Lib::template vector_t<N>;

    constexpr T alpha    = T{0.1};
    constexpr T h_space  = T{1} / T{N + 1};
    constexpr T coeff    = alpha / (h_space * h_space);
    constexpr T t_end    = T{0.01};
    constexpr T dt       = t_end / T{kSteps};
    constexpr T pi       = T{3.14159265358979323846};

    for (auto _ : state) {

        Mat L{};
        for (std::size_t i = 0; i < N; ++i) {
            Lib::Set(L, i, i, -T{2} * coeff);
            if (i > 0)     Lib::Set(L, i, i - 1, coeff);
            if (i + 1 < N) Lib::Set(L, i, i + 1, coeff);
        }

        Vec u{};
        for (std::size_t i = 0; i < N; ++i) {
            T xi = static_cast<T>(i + 1) * h_space;
            if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) u[i] = std::sin(pi * xi);
            else                                                   u(i) = std::sin(pi * xi);
        }

        for (int step = 0; step < kSteps; ++step) {
            Vec k1 = L * u;
            Vec k2 = L * Vec(u + k1 * (dt / T{2}));
            Vec k3 = L * Vec(u + k2 * (dt / T{2}));
            Vec k4 = L * Vec(u + k3 * dt);
            u = u + (k1 + k2 * T{2} + k3 * T{2} + k4) * (dt / T{6});
        }
        benchmark::DoNotOptimize(u);
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kSteps) * static_cast<int64_t>(8 * N * N));
}

}

#define REG_OP(BM, LIB, T)                                                                      \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,   4)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,   8)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  16)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  32)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  64)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>, 128)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>, 256)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_T(BM, T)  REG_OP(BM, LuminaTag, T); REG_OP(BM, EigenTag, T); REG_OP(BM, ArmaTag, T)

REG_T(BenchmarkHeatRk4, float);  REG_T(BenchmarkHeatRk4, double);
