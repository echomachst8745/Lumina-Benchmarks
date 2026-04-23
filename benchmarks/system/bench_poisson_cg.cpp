// Benchmarks a Poisson CG solve pipeline.

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>

#include "common/adapters.hpp"

namespace {

constexpr int kCGIters = 50;

template<class Lib, std::size_t N>
void BenchmarkPoissonCg(benchmark::State& state)
{
    using T   = typename Lib::scalar_t;
    using Mat = typename Lib::template matrix_t<N, N>;
    using Vec = typename Lib::template vector_t<N>;

    constexpr T h      = T{1} / T{N + 1};
    constexpr T inv_h2 = T{1} / (h * h);
    constexpr T pi     = T{3.14159265358979323846};

    for (auto _ : state) {

        Mat A{};
        for (std::size_t i = 0; i < N; ++i) {
            Lib::Set(A, i, i, T{2} * inv_h2);
            if (i > 0)     Lib::Set(A, i, i - 1, -inv_h2);
            if (i + 1 < N) Lib::Set(A, i, i + 1, -inv_h2);
        }
        Vec b{};
        for (std::size_t i = 0; i < N; ++i) {
            T xi = static_cast<T>(i + 1) * h;
            if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) b[i]  = std::sin(pi * xi);
            else                                                   b(i)  = std::sin(pi * xi);
        }

        Vec x{};
        Vec r  = b;
        Vec p  = r;
        Vec Ap;
        T   rr = Lib::Dot(r, r);
        for (int k = 0; k < kCGIters; ++k) {
            Ap          = A * p;
            T alpha     = rr / Lib::Dot(p, Ap);
            x           = x + p  * alpha;
            r           = r - Ap * alpha;
            T rr_new    = Lib::Dot(r, r);
            T beta      = rr_new / rr;
            p           = r + p * beta;
            rr          = rr_new;
        }
        benchmark::DoNotOptimize(x);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(
        static_cast<double>(kCGIters) * 2.0 * double{N} * N));
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

REG_T(BenchmarkPoissonCg, float);  REG_T(BenchmarkPoissonCg, double);
