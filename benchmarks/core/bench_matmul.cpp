// Benchmarks square matrix multiplication.

#include <benchmark/benchmark.h>

#include "common/adapters.hpp"
#include "common/generators.hpp"

namespace {

template<class Lib, std::size_t N>
void BenchmarkMatMul(benchmark::State& state)
{
    using Mat = typename Lib::template matrix_t<N, N>;

    auto A = bench::MakeRandomMatrix<Lib, N, N>(42);
    auto B = bench::MakeRandomMatrix<Lib, N, N>(43);
    Mat   C{};

    for (auto _ : state) {
        C = A * B;
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }

    const int64_t flops_per_iter = int64_t{2} * N * N * N - int64_t{N} * N;
    state.SetItemsProcessed(state.iterations() * flops_per_iter);
    state.SetBytesProcessed(state.iterations() *
        int64_t{3} * N * N * sizeof(typename Lib::scalar_t));
    state.SetLabel(std::string(Lib::kName) + "/N=" + std::to_string(N));
}

}

#define REGISTER_MATMUL_FOR_LIB(LIB, T)                                                \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>,   4)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>,   8)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>,  16)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>,  32)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>,  64)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>, 128)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BenchmarkMatMul, bench::LIB<T>, 256)->Repetitions(10)->ReportAggregatesOnly(true)

#define REGISTER_MATMUL_FOR_TYPE(T) \
    REGISTER_MATMUL_FOR_LIB(LuminaTag, T); \
    REGISTER_MATMUL_FOR_LIB(EigenTag,  T); \
    REGISTER_MATMUL_FOR_LIB(ArmaTag,   T)

REGISTER_MATMUL_FOR_TYPE(float);
REGISTER_MATMUL_FOR_TYPE(double);
