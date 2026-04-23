// Benchmarks dominant eigenvalue calculations.

#include <benchmark/benchmark.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/eigen_solvers.hpp>

namespace {

constexpr std::size_t kPowerIters = 50;

template<class Lib, std::size_t N>
void BenchmarkPowerIterHandRolled(benchmark::State& state)
{
    auto A  = bench::MakeSpdMatrix<Lib, N>(42);
    auto v0 = bench::MakeRandomVector<Lib, N>(99);
    using T = typename Lib::scalar_t;

    {
        T n0 = Lib::NormL2(v0);
        v0   = v0 * (T{1} / n0);
    }

    for (auto _ : state) {
        typename Lib::template vector_t<N> v = v0;
        typename Lib::template vector_t<N> Av;
        T lambda = T{0};
        for (std::size_t k = 0; k < kPowerIters; ++k) {
            Av         = A * v;
            T rayleigh = Lib::Dot(v, Av);
            T n        = Lib::NormL2(Av);
            v          = Av * (T{1} / n);
            lambda     = rayleigh;
        }
        benchmark::DoNotOptimize(lambda);
        benchmark::DoNotOptimize(v);
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kPowerIters * 2.0 * double{N} * N));
}

template<class Lib, std::size_t N>
void BenchmarkPowerIterNative(benchmark::State& state)
{
    auto A = bench::MakeSpdMatrix<Lib, N>(42);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) {
            auto r = lumina::linearalgebra::PowerIteration(A);
            benchmark::DoNotOptimize(r);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<T>>) {

            Eigen::SelfAdjointEigenSolver<typename Lib::template matrix_t<N, N>> es(A);
            T lambda_max = es.eigenvalues()(static_cast<Eigen::Index>(N) - 1);
            benchmark::DoNotOptimize(lambda_max);
        } else {
            arma::Col<T> eigval;
            arma::eig_sym(eigval, A);
            T lambda_max = eigval(N - 1);
            benchmark::DoNotOptimize(lambda_max);
        }
    }
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

REG_T(BenchmarkPowerIterHandRolled, float);  REG_T(BenchmarkPowerIterHandRolled, double);
REG_T(BenchmarkPowerIterNative,     float);  REG_T(BenchmarkPowerIterNative,     double);
