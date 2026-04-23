// Benchmarks Gaussian solve, inverse, and determinant.

#include <benchmark/benchmark.h>

#include <Eigen/LU>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/gaussian_elimination.hpp>

namespace {

template<class Lib, std::size_t N>
void BenchmarkGaussianSolve(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto x = lumina::linearalgebra::GaussianSolve(A, b);
            benchmark::DoNotOptimize(x);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {

            auto x = A.partialPivLu().solve(b).eval();
            benchmark::DoNotOptimize(x);
        } else {
            arma::Col<typename Lib::scalar_t> x = arma::solve(A, b);
            benchmark::DoNotOptimize(x);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>((2.0 / 3.0) * double{N} * N * N + 2.0 * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkInverse(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto inv = lumina::linearalgebra::Inverse(A);
            benchmark::DoNotOptimize(inv);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            typename Lib::template matrix_t<N, N> inv = A.inverse();
            benchmark::DoNotOptimize(inv);
        } else {
            typename Lib::template matrix_t<N, N> inv;
            inv = arma::inv(A);
            benchmark::DoNotOptimize(inv);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(2.0 * double{N} * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkDeterminant(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto d = lumina::linearalgebra::Determinant(A);
            benchmark::DoNotOptimize(d);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            auto d = A.determinant();
            benchmark::DoNotOptimize(d);
        } else {
            auto d = arma::det(A);
            benchmark::DoNotOptimize(d);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>((2.0 / 3.0) * double{N} * N * N));
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

REG_T(BenchmarkGaussianSolve, float);  REG_T(BenchmarkGaussianSolve, double);
REG_T(BenchmarkInverse,       float);  REG_T(BenchmarkInverse,       double);
REG_T(BenchmarkDeterminant,   float);  REG_T(BenchmarkDeterminant,   double);
