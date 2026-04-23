// Benchmarks LU decomposition and solve.

#include <benchmark/benchmark.h>

#include <Eigen/LU>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/lu_decomposition.hpp>

namespace {

template<class Lib, std::size_t N>
void BenchmarkLuDecompose(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto lu = lumina::linearalgebra::LUDecompose(A);
            benchmark::DoNotOptimize(lu);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            Eigen::PartialPivLU<typename Lib::template matrix_t<N, N>> lu(A);
            benchmark::DoNotOptimize(lu);
        } else {
            arma::Mat<typename Lib::scalar_t> L, U, P;
            arma::lu(L, U, P, A);
            benchmark::DoNotOptimize(L);
            benchmark::DoNotOptimize(U);
            benchmark::DoNotOptimize(P);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>((2.0 / 3.0) * double{N} * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkLuSolve(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto lu = lumina::linearalgebra::LUDecompose(A);
            auto x  = lumina::linearalgebra::LUSolve(lu, b);
            benchmark::DoNotOptimize(x);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            Eigen::PartialPivLU<typename Lib::template matrix_t<N, N>> lu(A);
            auto x = lu.solve(b).eval();
            benchmark::DoNotOptimize(x);
        } else {

            arma::Col<typename Lib::scalar_t> x = arma::solve(A, b);
            benchmark::DoNotOptimize(x);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>((2.0 / 3.0) * double{N} * N * N + 2.0 * N * N));
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

REG_T(BenchmarkLuDecompose, float);  REG_T(BenchmarkLuDecompose, double);
REG_T(BenchmarkLuSolve,     float);  REG_T(BenchmarkLuSolve,     double);
