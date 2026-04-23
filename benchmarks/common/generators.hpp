// Shared input generators for the benchmarks.

#pragma once

#include <cstddef>
#include <random>
#include <type_traits>

#include "adapters.hpp"

namespace bench {

template<class Lib, std::size_t N, std::size_t M>
typename Lib::template matrix_t<N, M> MakeRandomMatrix(std::uint64_t seed)
{
    using T = typename Lib::scalar_t;
    typename Lib::template matrix_t<N, M> out;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < M; ++j)
            Lib::Set(out, i, j, dist(rng));
    return out;
}

template<class Lib, std::size_t N>
typename Lib::template matrix_t<N, N> MakeSpdMatrix(std::uint64_t seed)
{
    using T = typename Lib::scalar_t;
    auto B = MakeRandomMatrix<Lib, N, N>(seed);
    typename Lib::template matrix_t<N, N> S;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T s{0};
            for (std::size_t k = 0; k < N; ++k)
                s += Lib::Get(B, k, i) * Lib::Get(B, k, j);
            if (i == j) s += static_cast<T>(N);
            Lib::Set(S, i, j, s);
        }
    }
    return S;
}

template<class Lib, std::size_t N>
typename Lib::template matrix_t<N, N> MakeDiagDominantMatrix(std::uint64_t seed)
{
    using T = typename Lib::scalar_t;
    auto A = MakeRandomMatrix<Lib, N, N>(seed);
    for (std::size_t i = 0; i < N; ++i) {
        T row_sum{0};
        for (std::size_t j = 0; j < N; ++j)
            if (i != j) row_sum += std::abs(Lib::Get(A, i, j));
        Lib::Set(A, i, i, row_sum + static_cast<T>(1));
    }
    return A;
}

template<class Lib, std::size_t N>
typename Lib::template vector_t<N> MakeRandomVector(std::uint64_t seed)
{
    using T = typename Lib::scalar_t;
    typename Lib::template vector_t<N> out;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});

    if constexpr (std::is_same_v<Lib, LuminaTag<T>>) {
        for (std::size_t i = 0; i < N; ++i) out[i] = dist(rng);
    } else {

        for (std::size_t i = 0; i < N; ++i) out(i) = dist(rng);
    }
    return out;
}

}
