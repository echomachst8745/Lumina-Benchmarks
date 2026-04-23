/**
 * @file lu_decomposition.hpp
 *
 * @brief LU decomposition with partial pivoting and associated solvers.
 *
 * @ingroup linearalgebra
 */
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <numeric>

#include <lumina/core/matrix.hpp>
#include <lumina/core/vector.hpp>

namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

/**
 * @brief Result of an LU decomposition: PA = LU.
 *
 * @tparam T Scalar element type.
 * @tparam N Matrix dimension.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
struct LUResult
{
    Matrix<T, N, N>            L;    ///< Lower-triangular factor (unit diagonal).
    Matrix<T, N, N>            U;    ///< Upper-triangular factor.
    std::array<std::size_t, N> perm; ///< Row permutation indices (index storage, not math).
    int                        sign; ///< Sign of the permutation (+1 or -1).
};

/**
 * @brief Computes the LU decomposition \f$PA = LU\f$ with partial pivoting. O(n³).
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Matrix dimension.
 *
 * @param A Square N×N matrix (passed by value).
 *
 * @return LUResult containing L, U, permutation, and sign.
 *
 * @throws std::runtime_error if the matrix is singular.
 *
 * @ingroup linearalgebra
 *
 * @code
 * lumina::core::Matrix3d A{{2.0, 1.0, 1.0},
 *                          {4.0, 3.0, 3.0},
 *                          {8.0, 7.0, 9.0}};
 * lumina::core::Vec3d b{1.0, 1.0, 1.0};
 * auto lu  = lumina::linearalgebra::LUDecompose(A);
 * auto x   = lumina::linearalgebra::LUSolve(lu, b);    // reuse for multiple b
 * auto det = lumina::linearalgebra::LUDeterminant(lu);
 * @endcode
 */
template<typename T, std::size_t N>
LUResult<T, N> LUDecompose(Matrix<T, N, N> A);

/**
 * @brief Solves Ax = b from a precomputed LU decomposition. O(n²).
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param lu Precomputed LUResult.
 * @param b  Right-hand side vector.
 *
 * @return Solution vector x.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const LUResult<T, N>& lu, Vector<T, N> b);

/**
 * @brief Convenience overload: decomposes A then solves Ax = b.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param A Square N×N matrix.
 * @param b Right-hand side vector.
 *
 * @return Solution vector x.
 *
 * @throws std::runtime_error if @p A is singular.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const Matrix<T, N, N>& A, Vector<T, N> b);

/**
 * @brief Computes det(A) from a precomputed LU decomposition.
 *
 * \f[
 *   \det(A) = \operatorname{sign}(P)\,\prod_{i=0}^{N-1} U_{ii}
 * \f]
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Matrix dimension.
 *
 * @param lu Precomputed LUResult from LUDecompose().
 *
 * @return Determinant of A.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
T LUDeterminant(const LUResult<T, N>& lu);

} // namespace lumina::linearalgebra

#include "../detail/linearalgebra/lu_decomposition.tpp"
