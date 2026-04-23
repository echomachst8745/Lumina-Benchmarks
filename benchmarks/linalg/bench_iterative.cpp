// Benchmarks Jacobi, Gauss-Seidel, and Conjugate Gradient solvers.

#include <benchmark/benchmark.h>

#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/iterative_solvers.hpp>

namespace {

constexpr int    kMaxIter = 50;
constexpr double kTolUnreachable = 1e-30;

template<class Lib, class Vec>
inline auto& VectorElement(Vec& v, std::size_t i)
{
    using T = typename Lib::scalar_t;
    if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) return v[i];
    else return v(i);
}
template<class Lib, class Vec>
inline auto ConstVectorElement(const Vec& v, std::size_t i)
{
    using T = typename Lib::scalar_t;
    if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) return v[i];
    else return v(i);
}

template<class Lib, std::size_t N>
void BenchmarkJacobiHandRolled(benchmark::State& state)
{
    auto A  = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b  = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        typename Lib::template vector_t<N> x{};
        typename Lib::template vector_t<N> x_new{};
        for (int k = 0; k < kMaxIter; ++k) {
            for (std::size_t i = 0; i < N; ++i) {
                T sigma{0};
                for (std::size_t j = 0; j < N; ++j)
                    if (i != j) sigma += Lib::Get(A, i, j) * ConstVectorElement<Lib>(x, j);
                VectorElement<Lib>(x_new, i) = (ConstVectorElement<Lib>(b, i) - sigma) / Lib::Get(A, i, i);
            }
            x = x_new;
        }
        benchmark::DoNotOptimize(x);
    }
    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kMaxIter) * static_cast<int64_t>(2 * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkJacobiNative(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        auto r = lumina::linearalgebra::Jacobi(A, b,
            typename Lib::template vector_t<N>{}, kMaxIter, T{kTolUnreachable});
        benchmark::DoNotOptimize(r);
    }
}

template<class Lib, std::size_t N>
void BenchmarkGaussSeidelHandRolled(benchmark::State& state)
{
    auto A  = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b  = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        typename Lib::template vector_t<N> x{};
        for (int k = 0; k < kMaxIter; ++k) {
            for (std::size_t i = 0; i < N; ++i) {
                T sigma{0};
                for (std::size_t j = 0; j < N; ++j)
                    if (i != j) sigma += Lib::Get(A, i, j) * ConstVectorElement<Lib>(x, j);
                VectorElement<Lib>(x, i) = (ConstVectorElement<Lib>(b, i) - sigma) / Lib::Get(A, i, i);
            }
        }
        benchmark::DoNotOptimize(x);
    }
    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kMaxIter) * static_cast<int64_t>(2 * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkGaussSeidelNative(benchmark::State& state)
{
    auto A = bench::MakeDiagDominantMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        auto r = lumina::linearalgebra::GaussSeidel(A, b,
            typename Lib::template vector_t<N>{}, kMaxIter, T{kTolUnreachable});
        benchmark::DoNotOptimize(r);
    }
}

template<class Lib, std::size_t N>
void BenchmarkCgHandRolled(benchmark::State& state)
{
    auto A  = bench::MakeSpdMatrix<Lib, N>(42);
    auto b  = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        using Vec = typename Lib::template vector_t<N>;
        Vec x{};
        Vec r  = b;
        Vec p  = r;
        Vec Ap;
        T   rr = Lib::Dot(r, r);
        for (int k = 0; k < kMaxIter; ++k) {
            Ap            = A * p;
            T    pAp      = Lib::Dot(p, Ap);
            T    alpha    = rr / pAp;
            x             = x + p  * alpha;
            r             = r - Ap * alpha;
            T    rr_new   = Lib::Dot(r, r);
            T    beta     = rr_new / rr;
            p             = r + p * beta;
            rr            = rr_new;
        }
        benchmark::DoNotOptimize(x);
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kMaxIter) * static_cast<int64_t>(2 * N * N));
}

template<class Lib, std::size_t N>
void BenchmarkCgNativeLumina(benchmark::State& state)
{
    auto A = bench::MakeSpdMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    for (auto _ : state) {
        auto r = lumina::linearalgebra::ConjugateGradient(A, b,
            typename Lib::template vector_t<N>{}, kMaxIter, T{kTolUnreachable});
        benchmark::DoNotOptimize(r);
    }
}

template<class Lib, std::size_t N>
void BenchmarkCgNativeEigen(benchmark::State& state)
{
    auto A = bench::MakeSpdMatrix<Lib, N>(42);
    auto b = bench::MakeRandomVector<Lib, N>(43);
    using T = typename Lib::scalar_t;

    Eigen::ConjugateGradient<
        typename Lib::template matrix_t<N, N>,
        Eigen::Lower | Eigen::Upper,
        Eigen::IdentityPreconditioner> cg;
    cg.setMaxIterations(kMaxIter);
    cg.setTolerance(T{kTolUnreachable});
    cg.compute(A);

    for (auto _ : state) {
        auto x = cg.solve(b).eval();
        benchmark::DoNotOptimize(x);
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

#define REG_T_ALL(BM, T)  REG_OP(BM, LuminaTag, T); REG_OP(BM, EigenTag, T); REG_OP(BM, ArmaTag, T)
#define REG_T_LUMINA(BM, T)  REG_OP(BM, LuminaTag, T)
#define REG_T_EIGEN(BM, T)   REG_OP(BM, EigenTag, T)

REG_T_ALL(BenchmarkJacobiHandRolled,      float);  REG_T_ALL(BenchmarkJacobiHandRolled,      double);
REG_T_ALL(BenchmarkGaussSeidelHandRolled, float);  REG_T_ALL(BenchmarkGaussSeidelHandRolled, double);
REG_T_ALL(BenchmarkCgHandRolled,          float);  REG_T_ALL(BenchmarkCgHandRolled,          double);

REG_T_LUMINA(BenchmarkJacobiNative,      float);  REG_T_LUMINA(BenchmarkJacobiNative,      double);
REG_T_LUMINA(BenchmarkGaussSeidelNative, float);  REG_T_LUMINA(BenchmarkGaussSeidelNative, double);
REG_T_LUMINA(BenchmarkCgNativeLumina,   float);  REG_T_LUMINA(BenchmarkCgNativeLumina,   double);
REG_T_EIGEN (BenchmarkCgNativeEigen,    float);  REG_T_EIGEN (BenchmarkCgNativeEigen,    double);
