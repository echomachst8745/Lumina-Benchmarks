// Benchmarks SuiteSparse matrix Cholesky and CG cases.

#include <benchmark/benchmark.h>

#include <Eigen/Cholesky>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/mtx_loader.hpp"

#include <lumina/linearalgebra/cholesky.hpp>

#ifndef LUMINA_BENCH_DATA_DIR
#define LUMINA_BENCH_DATA_DIR "data"
#endif

namespace {

constexpr int kCGIters = 50;

template<class Lib, std::size_t N>
const typename Lib::template matrix_t<N, N>&
LoadCachedMatrix(const char* path)
{
    using T = typename Lib::scalar_t;
    static const std::array<T, N * N> raw =
        bench::LoadSymmetricMtx<T, N>(path);
    static const auto M = bench::ToLibMatrix<Lib, N>(raw);
    return M;
}

template<class Lib, std::size_t N>
void RunSuiteSparseCholesky(benchmark::State& state, const char* path)
{
    using T = typename Lib::scalar_t;
    const auto& S = LoadCachedMatrix<Lib, N>(path);

    for (auto _ : state) {
        if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) {
            auto ch = lumina::linearalgebra::CholeskyDecompose(S);
            benchmark::DoNotOptimize(ch);
        } else if constexpr (std::is_same_v<Lib, bench::EigenTag<T>>) {
            Eigen::LLT<typename Lib::template matrix_t<N, N>> llt(S);
            benchmark::DoNotOptimize(llt);
        } else {
            typename Lib::template matrix_t<N, N> L;
            L = arma::chol(S, "lower");
            benchmark::DoNotOptimize(L);
        }
    }

    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(double{N} * N * N / 3.0));
}

template<class Lib, std::size_t N>
void RunSuiteSparseCg(benchmark::State& state, const char* path)
{
    using T   = typename Lib::scalar_t;
    using Vec = typename Lib::template vector_t<N>;

    const auto& A = LoadCachedMatrix<Lib, N>(path);

    static const Vec b = [] {
        Vec v{};
        for (std::size_t i = 0; i < N; ++i) {
            if constexpr (std::is_same_v<Lib, bench::LuminaTag<T>>) v[i] = T{1};
            else                                                   v(i) = T{1};
        }
        return v;
    }();

    for (auto _ : state) {
        Vec x{};
        Vec r = b;
        Vec p = r;
        Vec Ap;
        T   rr = Lib::Dot(r, r);
        for (int k = 0; k < kCGIters; ++k) {
            Ap          = A * p;
            T pAp       = Lib::Dot(p, Ap);
            T alpha     = rr / pAp;
            x           = x + p  * alpha;
            r           = r - Ap * alpha;
            T rr_new    = Lib::Dot(r, r);
            T beta      = rr_new / rr;
            p           = r + p * beta;
            rr          = rr_new;
        }
        benchmark::DoNotOptimize(x);
    }
    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(kCGIters) * static_cast<int64_t>(2 * N * N));
}

}

#define SS_DATA_PATH(NAME) (LUMINA_BENCH_DATA_DIR "/suitesparse/" NAME ".mtx")

#define DEFINE_SS_FUNCS(NAME)                                                 \
    template<class Lib, std::size_t N>                                        \
    static void BenchmarkSuiteSparseCholesky##NAME(benchmark::State& s)            \
    { RunSuiteSparseCholesky<Lib, N>(s, SS_DATA_PATH(#NAME)); }                      \
    template<class Lib, std::size_t N>                                        \
    static void BenchmarkSuiteSparseCg##NAME(benchmark::State& s)                  \
    { RunSuiteSparseCg<Lib, N>(s, SS_DATA_PATH(#NAME)); }

#define REG_SS_ALL(NAME, DIM)                                                                               \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCholesky##NAME, bench::LuminaTag<double>, DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCholesky##NAME, bench::EigenTag<double>,  DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCholesky##NAME, bench::ArmaTag<double>,   DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCg##NAME,       bench::LuminaTag<double>, DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCg##NAME,       bench::EigenTag<double>,  DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCg##NAME,       bench::ArmaTag<double>,   DIM)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_SS_NOLUMINA(NAME, DIM)                                                                          \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCholesky##NAME, bench::EigenTag<double>,  DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCholesky##NAME, bench::ArmaTag<double>,   DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCg##NAME,       bench::EigenTag<double>,  DIM)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkSuiteSparseCg##NAME,       bench::ArmaTag<double>,   DIM)->Repetitions(10)->ReportAggregatesOnly(true)

DEFINE_SS_FUNCS(bcsstk01)
DEFINE_SS_FUNCS(mesh1e1)
DEFINE_SS_FUNCS(nos4)
DEFINE_SS_FUNCS(bcsstk03)
DEFINE_SS_FUNCS(bcsstk04)
DEFINE_SS_FUNCS(bcsstk05)

REG_SS_ALL(bcsstk01,  48);
REG_SS_ALL(mesh1e1,   48);
REG_SS_ALL(nos4,     100);
REG_SS_ALL(bcsstk03, 112);
REG_SS_ALL(bcsstk04, 132);
REG_SS_ALL(bcsstk05, 153);
