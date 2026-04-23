// Adapter types for Lumina, Eigen, and Armadillo benchmarks.

#pragma once

#include <cstddef>
#include <string_view>

#include <lumina/core/matrix.hpp>
#include <lumina/core/vector.hpp>

#include <Eigen/Dense>

#include <armadillo>

namespace bench {

template<class T>
struct LuminaTag
{
    using scalar_t = T;

    template<std::size_t N, std::size_t M>
    using matrix_t = lumina::core::Matrix<T, N, M>;

    template<std::size_t N>
    using vector_t = lumina::core::Vector<T, N>;

    static constexpr std::string_view kName = "Lumina";

    template<class Mat>
    static void Set(Mat& mat, std::size_t i, std::size_t j, T value) noexcept
    {
        mat(i, j) = value;
    }

    template<class Mat>
    static T Get(const Mat& mat, std::size_t i, std::size_t j) noexcept
    {
        return mat(i, j);
    }

    template<class Mat>  static auto Transpose(const Mat& m)               { return m.Transpose(); }
    template<class Vec>  static auto Dot(const Vec& a, const Vec& b)       { return a.Dot(b); }
    template<class Vec>  static auto NormL2(const Vec& v)                 { return v.Norm(); }
    template<class Vec>  static auto NormL1(const Vec& v)                 { return v.NormL1(); }
    template<class Vec>  static auto NormLInf(const Vec& v)               { return v.NormLInf(); }
    template<class Vec>  static auto Sum(const Vec& v)                     { return v.Sum(); }
    template<class Mat>  static auto Trace(const Mat& m)                   { return m.Trace(); }
    template<class Mat>  static auto Frobenius(const Mat& m)               { return m.FrobeniusNorm(); }
};

template<class T>
struct EigenTag
{
    using scalar_t = T;

    template<std::size_t N, std::size_t M>
    using matrix_t = Eigen::Matrix<T, static_cast<int>(N), static_cast<int>(M)>;

    template<std::size_t N>
    using vector_t = Eigen::Matrix<T, static_cast<int>(N), 1>;

    static constexpr std::string_view kName = "Eigen";

    template<class Mat>
    static void Set(Mat& mat, std::size_t i, std::size_t j, T value) noexcept
    {
        mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
    }

    template<class Mat>
    static T Get(const Mat& mat, std::size_t i, std::size_t j) noexcept
    {
        return mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    }

    template<class Mat>  static auto Transpose(const Mat& m)               { return m.transpose().eval(); }
    template<class Vec>  static auto Dot(const Vec& a, const Vec& b)       { return a.dot(b); }
    template<class Vec>  static auto NormL2(const Vec& v)                 { return v.norm(); }
    template<class Vec>  static auto NormL1(const Vec& v)                 { return v.template lpNorm<1>(); }
    template<class Vec>  static auto NormLInf(const Vec& v)               { return v.template lpNorm<Eigen::Infinity>(); }
    template<class Vec>  static auto Sum(const Vec& v)                     { return v.sum(); }
    template<class Mat>  static auto Trace(const Mat& m)                   { return m.trace(); }
    template<class Mat>  static auto Frobenius(const Mat& m)               { return m.norm(); }
};

template<class T>
struct ArmaTag
{
    using scalar_t = T;

    template<std::size_t N, std::size_t M>
    using matrix_t = typename arma::Mat<T>::template fixed<N, M>;

    template<std::size_t N>
    using vector_t = typename arma::Col<T>::template fixed<N>;

    static constexpr std::string_view kName = "Armadillo";

    template<class Mat>
    static void Set(Mat& mat, std::size_t i, std::size_t j, T value) noexcept
    {
        mat(i, j) = value;
    }

    template<class Mat>
    static T Get(const Mat& mat, std::size_t i, std::size_t j) noexcept
    {
        return mat(i, j);
    }

    template<class Mat>  static auto Transpose(const Mat& m)               { Mat r; r = m.t(); return r; }
    template<class Vec>  static auto Dot(const Vec& a, const Vec& b)       { return arma::dot(a, b); }
    template<class Vec>  static auto NormL2(const Vec& v)                 { return arma::norm(v, 2); }
    template<class Vec>  static auto NormL1(const Vec& v)                 { return arma::norm(v, 1); }
    template<class Vec>  static auto NormLInf(const Vec& v)               { return arma::norm(v, "inf"); }
    template<class Vec>  static auto Sum(const Vec& v)                     { return arma::accu(v); }
    template<class Mat>  static auto Trace(const Mat& m)                   { return arma::trace(m); }
    template<class Mat>  static auto Frobenius(const Mat& m)               { return arma::norm(m, "fro"); }
};

}
