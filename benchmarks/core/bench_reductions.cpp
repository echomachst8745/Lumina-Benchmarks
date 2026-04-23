// Benchmarks trace, Frobenius norm, and vector sum.

#include <benchmark/benchmark.h>

#include "common/adapters.hpp"
#include "common/generators.hpp"

namespace {

template<class Lib, std::size_t N>
void BenchmarkTrace(benchmark::State& state)
{
    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    for (auto _ : state) {
        auto r = Lib::Trace(A);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * int64_t{N});
    state.SetBytesProcessed(state.iterations() * int64_t{N} * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

template<class Lib, std::size_t N>
void BenchmarkFrobenius(benchmark::State& state)
{
    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    for (auto _ : state) {
        auto r = Lib::Frobenius(A);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * int64_t{N} * N);
    state.SetBytesProcessed(state.iterations() *
        int64_t{N} * N * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

template<class Lib, std::size_t N>
void BenchmarkVecSum(benchmark::State& state)
{
    auto v = bench::MakeRandomVector<Lib, N>(42);
    for (auto _ : state) {
        auto r = Lib::Sum(v);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * int64_t{N});
    state.SetBytesProcessed(state.iterations() * int64_t{N} * sizeof(typename Lib::scalar_t));
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

REG_T(BenchmarkTrace,     float);  REG_T(BenchmarkTrace,     double);
REG_T(BenchmarkFrobenius, float);  REG_T(BenchmarkFrobenius, double);
REG_T(BenchmarkVecSum,    float);  REG_T(BenchmarkVecSum,    double);
