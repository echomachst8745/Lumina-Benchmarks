// Benchmarks Cholesky decomposition and solve.

#include <benchmark/benchmark.h>

#include <Eigen/Cholesky>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/cholesky.hpp>

namespace {

template<class Lib, std::size_t N>
void BenchmarkCholeskyDecompose(benchmark::State& state)
{
    auto A = bench::MakeSpdMatrix<Lib, N>(42);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto ch = lumina::linearalgebra::CholeskyDecompose(A);
            benchmark::DoNotOptimize(ch);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            Eigen::LLT<typename Lib::template matrix_t<N, N>> llt(A);
            benchmark::DoNotOptimize(llt);
        } else {
            typename Lib::template matrix_t<N, N> L;
            L = arma::chol(A, "lower");
            benchmark::DoNotOptimize(L);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(double{N} * N * N / 3.0));
}

template<class Lib, std::size_t N>
void BenchmarkCholeskySolve(benchmark::State& state)
{
    auto A = bench::MakeSpdMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<typename Lib::scalar_t>>) {
            auto x = lumina::linearalgebra::CholeskySolve(A, b);
            benchmark::DoNotOptimize(x);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<typename Lib::scalar_t>>) {
            Eigen::LLT<typename Lib::template matrix_t<N, N>> llt(A);
            auto x = llt.solve(b).eval();
            benchmark::DoNotOptimize(x);
        } else {

            arma::Col<typename Lib::scalar_t> x =
                arma::solve(A, b, arma::solve_opts::likely_sympd);
            benchmark::DoNotOptimize(x);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(double{N} * N * N / 3.0 + 2.0 * N * N));
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

REG_T(BenchmarkCholeskyDecompose, float);  REG_T(BenchmarkCholeskyDecompose, double);
REG_T(BenchmarkCholeskySolve,     float);  REG_T(BenchmarkCholeskySolve,     double);
