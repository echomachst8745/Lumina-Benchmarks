// Benchmarks RK4 integration on the Lorenz system.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <type_traits>
#include <vector>

#include "common/adapters.hpp"

#include <lumina/ode/runge_kutta.hpp>

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

template<class Lib, class Vec>
inline Vec LorenzRhs(const Vec& y)
{
    using T = typename Lib::scalar_t;
    constexpr T sigma = T{10};
    constexpr T rho   = T{28};
    constexpr T beta  = T{8} / T{3};
    Vec dy{};
    const T x = ConstVectorElement<Lib>(y, 0);
    const T z = ConstVectorElement<Lib>(y, 1);
    const T w = ConstVectorElement<Lib>(y, 2);
    VectorElement<Lib>(dy, 0) = sigma * (z - x);
    VectorElement<Lib>(dy, 1) = x * (rho - w) - z;
    VectorElement<Lib>(dy, 2) = x * z - beta * w;
    return dy;
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkRk4HandRolled(benchmark::State& state)
{
    using T  = typename Lib::scalar_t;
    using V  = typename Lib::template vector_t<3>;
    constexpr T h = T{1e-3};

    for (auto _ : state) {
        V y{};
        VectorElement<Lib>(y, 0) = T{1};
        VectorElement<Lib>(y, 1) = T{1};
        VectorElement<Lib>(y, 2) = T{1};
        for (std::size_t k = 0; k < N_STEPS; ++k) {
            V k1 = LorenzRhs<Lib, V>(y);
            V k2 = LorenzRhs<Lib, V>(static_cast<V>(y + k1 * (h / T{2})));
            V k3 = LorenzRhs<Lib, V>(static_cast<V>(y + k2 * (h / T{2})));
            V k4 = LorenzRhs<Lib, V>(static_cast<V>(y + k3 * h));
            y    = y + (k1 + k2 * T{2} + k3 * T{2} + k4) * (h / T{6});
        }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkRk4Native(benchmark::State& state)
{
    static_assert(std::is_same_v<Lib, bench::LuminaTag<double>>,
                  "Native ODE is Lumina-only");
    constexpr double h = 1e-3;
    lumina::ode::ODEFunction f = [](double , const lumina::ode::State& y) {
        constexpr double sigma = 10.0;
        constexpr double rho   = 28.0;
        constexpr double beta  = 8.0 / 3.0;
        return lumina::ode::State{
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]
        };
    };
    for (auto _ : state) {
        auto r = lumina::ode::RK4(f, 0.0,
                                  double{N_STEPS} * h,
                                  lumina::ode::State{1.0, 1.0, 1.0}, h);
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

REG_ALL(BenchmarkRk4HandRolled);
REG_LUMINA(BenchmarkRk4Native);
