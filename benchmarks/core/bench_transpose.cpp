// Benchmarks square matrix transposition.

#include <benchmark/benchmark.h>

#include "common/adapters.hpp"
#include "common/generators.hpp"

namespace {

template<class Lib, std::size_t N>
void BenchmarkTranspose(benchmark::State& state)
{
    using Mat = typename Lib::template matrix_t<N, N>;

    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    Mat  B{};

    for (auto _ : state) {
        B = Lib::Transpose(A);
        benchmark::DoNotOptimize(B);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * int64_t{N} * N);
    state.SetBytesProcessed(state.iterations() *
        int64_t{2} * N * N * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

}

#define REG(LIB, T)                                                                              \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>,   4)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>,   8)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>,  16)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>,  32)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>,  64)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>, 128)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkTranspose, bench::LIB<T>, 256)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_T(T) REG(LuminaTag, T); REG(EigenTag, T); REG(ArmaTag, T)

REG_T(float);
REG_T(double);
