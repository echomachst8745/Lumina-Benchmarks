// Benchmarks matrix addition and scalar multiplication.

#include <benchmark/benchmark.h>

#include "common/adapters.hpp"
#include "common/generators.hpp"

namespace {

template<class Lib, std::size_t N>
void BenchmarkMatAdd(benchmark::State& state)
{
    using Mat = typename Lib::template matrix_t<N, N>;

    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    auto B = bench::MakeRandomMatrix<Lib, N, N>(43);
    Mat  C{};

    for (auto _ : state) {
        C = A + B;
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * int64_t{N} * N);
    state.SetBytesProcessed(state.iterations() *
        int64_t{3} * N * N * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

template<class Lib, std::size_t N>
void BenchmarkScalarMul(benchmark::State& state)
{
    using Mat = typename Lib::template matrix_t<N, N>;
    using T   = typename Lib::scalar_t;

    auto A     = bench::MakeRandomMatrix<Lib, N, N>(42);
    T    alpha = T{1.7};
    Mat  C{};

    for (auto _ : state) {
        C = A * alpha;
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * int64_t{N} * N);
    state.SetBytesProcessed(state.iterations() *
        int64_t{2} * N * N * sizeof(T));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

}

#define REG_OP(BM, LIB, T)                                                                          \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,   4)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,   8)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  16)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  32)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>,  64)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>, 128)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<T>, 256)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_T(BM, T)  REG_OP(BM, LuminaTag, T); REG_OP(BM, EigenTag, T); REG_OP(BM, ArmaTag, T)

REG_T(BenchmarkMatAdd,    float);
REG_T(BenchmarkMatAdd,    double);
REG_T(BenchmarkScalarMul, float);
REG_T(BenchmarkScalarMul, double);
