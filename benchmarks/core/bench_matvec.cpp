// Benchmarks matrix-vector multiplication.

#include <benchmark/benchmark.h>

#include "common/adapters.hpp"
#include "common/generators.hpp"

namespace {

template<class Lib, std::size_t N>
void BenchmarkMatVec(benchmark::State& state)
{
    using Vec = typename Lib::template vector_t<N>;

    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    auto x = bench::MakeRandomVector<Lib, N>(43);
    Vec  y{};

    for (auto _ : state) {
        y = A * x;
        benchmark::DoNotOptimize(y);
        benchmark::ClobberMemory();
    }

    const int64_t flops_per_iter = int64_t{2} * N * N - int64_t{N};
    state.SetItemsProcessed(state.iterations() * flops_per_iter);
    state.SetBytesProcessed(state.iterations() *
        (int64_t{N} * N + int64_t{2} * N) * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

}

#define REG(LIB, T)                                                                          \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>,   4)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>,   8)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>,  16)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>,  32)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>,  64)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>, 128)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatVec, bench::LIB<T>, 256)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_T(T) REG(LuminaTag, T); REG(EigenTag, T); REG(ArmaTag, T)

REG_T(float);
REG_T(double);
