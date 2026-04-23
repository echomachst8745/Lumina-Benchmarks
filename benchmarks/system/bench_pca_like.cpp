// Benchmarks a PCA-style linear algebra pipeline.

#include <benchmark/benchmark.h>

#include <Eigen/Cholesky>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/cholesky.hpp>

namespace {

constexpr std::size_t kPowerIters = 50;

template<class Lib, std::size_t N>
void BenchmarkPcaPipeline(benchmark::State& state)
{
    using T    = typename Lib::scalar_t;
    constexpr std::size_t M = 2 * N;
    using MatMN = typename Lib::template matrix_t<M, N>;
    using MatNN = typename Lib::template matrix_t<N, N>;
    using VecN  = typename Lib::template vector_t<N>;

    constexpr T eps    = T{1e-3};
    constexpr T inv_M  = T{1} / T{M};

    auto X = bench::MakeRandomMatrix<Lib, M, N>(42);
    auto v0 = bench::MakeRandomVector<Lib, N>(99);
    {
        T n0 = Lib::NormL2(v0);
        v0   = v0 * (T{1} / n0);
    }

    for (auto _ : state) {

        MatNN S;
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) {
            S = X.Transpose() * X;
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<T>>) {
            S = (X.transpose() * X).eval();
        } else {
            S = X.t() * X;
        }
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                T s = Lib::Get(S, i, j) * inv_M + (i == j ? eps : T{0});
                Lib::Set(S, i, j, s);
            }

        if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) {
            auto ch = lumina::linearalgebra::CholeskyDecompose(S);
            benchmark::DoNotOptimize(ch);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<T>>) {
            Eigen::LLT<MatNN> llt(S);
            benchmark::DoNotOptimize(llt);
        } else {
            MatNN Lchol;
            Lchol = arma::chol(S, "lower");
            benchmark::DoNotOptimize(Lchol);
        }

        VecN v = v0;
        VecN Sv;
        T lambda = T{0};
        for (std::size_t k = 0; k < kPowerIters; ++k) {
            Sv         = S * v;
            T rq       = Lib::Dot(v, Sv);
            T n        = Lib::NormL2(Sv);
            v          = Sv * (T{1} / n);
            lambda     = rq;
        }
        benchmark::DoNotOptimize(lambda);
        benchmark::DoNotOptimize(v);
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(2.0 * double{M} * N * N));
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

REG_T(BenchmarkPcaPipeline, float);  REG_T(BenchmarkPcaPipeline, double);
