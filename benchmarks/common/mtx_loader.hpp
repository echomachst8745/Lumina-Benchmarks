// Matrix Market loading helpers for SuiteSparse benchmarks.

#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bench {

template<class T, std::size_t N>
std::array<T, N * N> LoadSymmetricMtx(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string header;
    if (!std::getline(in, header))
        throw std::runtime_error("empty file: " + path);

    if (header.rfind("%%MatrixMarket", 0) != 0 ||
        header.find("coordinate") == std::string::npos ||
        header.find("real")       == std::string::npos ||
        header.find("symmetric")  == std::string::npos)
    {
        throw std::runtime_error("unsupported MM header: " + header);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    std::size_t rows = 0, cols = 0, nnz = 0;
    {
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        if (!iss) throw std::runtime_error("bad size line in " + path);
    }
    if (rows != N || cols != N)
        throw std::runtime_error("dimension mismatch: file is " +
            std::to_string(rows) + "x" + std::to_string(cols) +
            ", expected " + std::to_string(N));

    std::array<T, N * N> M{};
    for (std::size_t k = 0; k < nnz; ++k) {
        std::size_t i = 0, j = 0;
        double v = 0.0;
        if (!(in >> i >> j >> v))
            throw std::runtime_error("bad triple in " + path);
        if (i == 0 || j == 0 || i > N || j > N)
            throw std::runtime_error("index out of range in " + path);
        --i; --j;
        M[i * N + j] = static_cast<T>(v);
        if (i != j) M[j * N + i] = static_cast<T>(v);
    }
    return M;
}

template<class Lib, std::size_t N>
typename Lib::template matrix_t<N, N>
ToLibMatrix(const std::array<typename Lib::scalar_t, N * N>& src)
{
    typename Lib::template matrix_t<N, N> out{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            Lib::Set(out, i, j, src[i * N + j]);
    return out;
}

}
