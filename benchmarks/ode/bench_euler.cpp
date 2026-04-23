// Benchmarks Euler ODE integrators.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <type_traits>
#include <vector>

#include "common/adapters.hpp"

#include <lumina/ode/euler.hpp>

namespace {

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

template<class Lib, std::size_t N_STEPS>
void BenchmarkEulerExplicitHandRolled(benchmark::State& state)
{
    using T  = typename Lib::scalar_t;
    using V  = typename Lib::template vector_t<2>;
    constexpr T h = T{1e-3};

    for (auto _ : state) {
        V y{};
        VectorElement<Lib>(y, 0) = T{1};
        VectorElement<Lib>(y, 1) = T{0};
        for (std::size_t k = 0; k < N_STEPS; ++k) {
            V dy{};
            VectorElement<Lib>(dy, 0) =  ConstVectorElement<Lib>(y, 1);
            VectorElement<Lib>(dy, 1) = -ConstVectorElement<Lib>(y, 0);
            y = y + dy * h;
        }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkEulerExplicitNative(benchmark::State& state)
{
    static_assert(std::is_same_v<Lib, bench::LuminaTag<double>>,
                  "Native ODE is Lumina-only (Eigen/Arma ship no integrators)");
    constexpr double h = 1e-3;
    lumina::ode::ODEFunction f = [](double , const lumina::ode::State& y) {
        return lumina::ode::State{ y[1], -y[0] };
    };
    for (auto _ : state) {
        auto r = lumina::ode::Euler(f, 0.0, double{N_STEPS} * h,
                                    lumina::ode::State{1.0, 0.0}, h);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkEulerImplicitHandRolled(benchmark::State& state)
{
    using T  = typename Lib::scalar_t;
    using V  = typename Lib::template vector_t<1>;
    constexpr T h         = T{1e-3};
    constexpr T lambda    = T{-50};
    constexpr int kMaxIt  = 100;
    constexpr T delta_tol = T{1e-10};

    for (auto _ : state) {
        V y{};
        VectorElement<Lib>(y, 0) = T{1};
        for (std::size_t k = 0; k < N_STEPS; ++k) {
            V yn = y;
            for (int it = 0; it < kMaxIt; ++it) {
                V yn_new{};
                VectorElement<Lib>(yn_new, 0) = ConstVectorElement<Lib>(y, 0) +
                                       h * lambda * ConstVectorElement<Lib>(yn, 0);
                T diff = ConstVectorElement<Lib>(yn_new, 0) - ConstVectorElement<Lib>(yn, 0);
                yn = yn_new;
                if (diff < T{0}) diff = -diff;
                if (diff < delta_tol) break;
            }
            y = yn;
        }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkEulerImplicitNative(benchmark::State& state)
{
    static_assert(std::is_same_v<Lib, bench::LuminaTag<double>>,
                  "Native ODE is Lumina-only");
    constexpr double h = 1e-3;
    lumina::ode::ODEFunction f = [](double , const lumina::ode::State& y) {
        return lumina::ode::State{ -50.0 * y[0] };
    };
    for (auto _ : state) {
        auto r = lumina::ode::EulerImplicit(f, 0.0,
                                            double{N_STEPS} * h,
                                            lumina::ode::State{1.0}, h);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

}

#define REG_OP(BM, LIB)                                                                              \
    BENCHMARK_TEMPLATE(BM, bench::LIB<double>,  1024)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<double>,  4096)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<double>, 16384)->Repetitions(10)->ReportAggregatesOnly(true); \
    BENCHMARK_TEMPLATE(BM, bench::LIB<double>, 65536)->Repetitions(10)->ReportAggregatesOnly(true)

#define REG_ALL(BM)     REG_OP(BM, LuminaTag); REG_OP(BM, EigenTag); REG_OP(BM, ArmaTag)
#define REG_LUMINA(BM)  REG_OP(BM, LuminaTag)

REG_ALL(BenchmarkEulerExplicitHandRolled);
REG_ALL(BenchmarkEulerImplicitHandRolled);
REG_LUMINA(BenchmarkEulerExplicitNative);
REG_LUMINA(BenchmarkEulerImplicitNative);
