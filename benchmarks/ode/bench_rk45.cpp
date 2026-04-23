// Benchmarks adaptive RK45 integration.

#include <benchmark/benchmark.h>

#include <cmath>
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
inline Vec VanDerPolRhs(const Vec& y)
{
    using T = typename Lib::scalar_t;
    constexpr T mu = T{5};
    Vec dy{};
    const T x  = ConstVectorElement<Lib>(y, 0);
    const T yv = ConstVectorElement<Lib>(y, 1);
    VectorElement<Lib>(dy, 0) = yv;
    VectorElement<Lib>(dy, 1) = mu * (T{1} - x * x) * yv - x;
    return dy;
}

template<class T> struct DP
{

    static constexpr T c2 = T{1} / T{5};
    static constexpr T c3 = T{3} / T{10};
    static constexpr T c4 = T{4} / T{5};
    static constexpr T c5 = T{8} / T{9};

    static constexpr T a21 = T{1} / T{5};
    static constexpr T a31 = T{3} / T{40};
    static constexpr T a32 = T{9} / T{40};
    static constexpr T a41 = T{44} / T{45};
    static constexpr T a42 = -T{56} / T{15};
    static constexpr T a43 = T{32} / T{9};
    static constexpr T a51 = T{19372} / T{6561};
    static constexpr T a52 = -T{25360} / T{2187};
    static constexpr T a53 = T{64448} / T{6561};
    static constexpr T a54 = -T{212} / T{729};
    static constexpr T a61 = T{9017} / T{3168};
    static constexpr T a62 = -T{355} / T{33};
    static constexpr T a63 = T{46732} / T{5247};
    static constexpr T a64 = T{49} / T{176};
    static constexpr T a65 = -T{5103} / T{18656};
    static constexpr T a71 = T{35} / T{384};
    static constexpr T a73 = T{500} / T{1113};
    static constexpr T a74 = T{125} / T{192};
    static constexpr T a75 = -T{2187} / T{6784};
    static constexpr T a76 = T{11} / T{84};

    static constexpr T e1 = T{71} / T{57600};
    static constexpr T e3 = -T{71} / T{16695};
    static constexpr T e4 = T{71} / T{1920};
    static constexpr T e5 = -T{17253} / T{339200};
    static constexpr T e6 = T{22} / T{525};
    static constexpr T e7 = -T{1} / T{40};
};

template<class Lib, std::size_t N_STEPS>
void BenchmarkRk45HandRolled(benchmark::State& state)
{
    using T  = typename Lib::scalar_t;
    using V  = typename Lib::template vector_t<2>;
    using C  = DP<T>;

    constexpr T t_end  = T{N_STEPS} * T{1e-3};
    constexpr T rtol   = T{1e-6};
    constexpr T atol   = T{1e-9};
    constexpr T safety = T{0.9};
    constexpr T min_f  = T{0.1};
    constexpr T max_f  = T{5};

    for (auto _ : state) {
        V y{};
        VectorElement<Lib>(y, 0) = T{2};
        VectorElement<Lib>(y, 1) = T{0};

        T t = T{0};
        T h = T{1e-2};
        V k1 = VanDerPolRhs<Lib, V>(y);

        while (t < t_end) {
            if (t + h > t_end) h = t_end - t;

            V k2 = VanDerPolRhs<Lib, V>(V(y + k1 * (C::a21 * h)));
            V k3 = VanDerPolRhs<Lib, V>(V(y + (k1 * C::a31 + k2 * C::a32) * h));
            V k4 = VanDerPolRhs<Lib, V>(V(y + (k1 * C::a41 + k2 * C::a42 + k3 * C::a43) * h));
            V k5 = VanDerPolRhs<Lib, V>(V(y + (k1 * C::a51 + k2 * C::a52 + k3 * C::a53
                                        + k4 * C::a54) * h));
            V k6 = VanDerPolRhs<Lib, V>(V(y + (k1 * C::a61 + k2 * C::a62 + k3 * C::a63
                                        + k4 * C::a64 + k5 * C::a65) * h));

            V y_new = y + (k1 * C::a71 + k3 * C::a73 + k4 * C::a74
                         + k5 * C::a75 + k6 * C::a76) * h;
            V k7    = VanDerPolRhs<Lib, V>(y_new);

            V err_v = (k1 * C::e1 + k3 * C::e3 + k4 * C::e4
                     + k5 * C::e5 + k6 * C::e6 + k7 * C::e7) * h;

            T err = T{0};
            for (std::size_t i = 0; i < 2; ++i) {
                T yi   = ConstVectorElement<Lib>(y, i);
                T yni  = ConstVectorElement<Lib>(y_new, i);
                T sc   = atol + rtol * std::max(std::abs(yi), std::abs(yni));
                T comp = std::abs(ConstVectorElement<Lib>(err_v, i)) / sc;
                if (comp > err) err = comp;
            }

            if (err <= T{1}) {
                t += h;
                y  = y_new;
                k1 = k7;
            }
            T fac = (err == T{0})
                  ? max_f
                  : safety * std::pow(T{1} / err, T{1} / T{5});
            if (fac < min_f) fac = min_f;
            if (fac > max_f) fac = max_f;
            h *= fac;
        }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N_STEPS));
}

template<class Lib, std::size_t N_STEPS>
void BenchmarkRk45Native(benchmark::State& state)
{
    static_assert(std::is_same_v<Lib, bench::LuminaTag<double>>,
                  "Native ODE is Lumina-only");
    constexpr double t_end = double{N_STEPS} * 1e-3;
    lumina::ode::ODEFunction f = [](double , const lumina::ode::State& y) {
        constexpr double mu = 5.0;
        return lumina::ode::State{
            y[1],
            mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
        };
    };
    for (auto _ : state) {
        auto r = lumina::ode::RK45(f, 0.0, t_end,
                                   lumina::ode::State{2.0, 0.0},
                                   1e-6, 1e-9, 1e-2);
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

REG_ALL(BenchmarkRk45HandRolled);
REG_LUMINA(BenchmarkRk45Native);
