/**
 * @file cholesky.hpp
 *
 * @brief Cholesky decomposition for symmetric positive-definite matrices.
 *
 * @ingroup linearalgebra
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <limits>

#include <lumina/core/matrix.hpp>
#include <lumina/core/vector.hpp>

namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

/**
 * @brief Result of a Cholesky decomposition: A = LL^T.
 *
 * @tparam T Scalar element type.
 * @tparam N Matrix dimension.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
struct CholeskyResult
{
    Matrix<T, N, N> L; ///< Lower-triangular Cholesky factor (A = LL^T).
};

/**
 * @brief Computes the Cholesky decomposition \f$A = LL^T\f$. O(n³/3).
 *
 * Uses the Cholesky-Banachiewicz algorithm (row-by-row). Roughly twice as
 * fast as LU decomposition for symmetric positive-definite (SPD) matrices
 * because it exploits symmetry and needs no pivoting.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Matrix dimension.
 *
 * @param A Square N×N symmetric positive-definite matrix (passed by value).
 *
 * @return CholeskyResult containing lower-triangular factor L.
 *
 * @throws std::runtime_error if the matrix is not positive-definite
 *         (i.e. a diagonal entry of L would be zero or imaginary).
 *
 * @ingroup linearalgebra
 *
 * @code
 * lumina::core::Matrix3d A{{4.0, 2.0, 2.0},
 *                          {2.0, 5.0, 3.0},
 *                          {2.0, 3.0, 6.0}};
 * auto ch  = lumina::linearalgebra::CholeskyDecompose(A);
 * auto x   = lumina::linearalgebra::CholeskySolve(ch, b);
 * auto det = lumina::linearalgebra::CholeskyDeterminant(ch);
 * @endcode
 */
template<typename T, std::size_t N>
CholeskyResult<T, N> CholeskyDecompose(Matrix<T, N, N> A);

/**
 * @brief Solves Ax = b from a precomputed Cholesky decomposition. O(n²).
 *
 * Performs two triangular solves:
 *   1. Forward substitution:  Ly  = b
 *   2. Back substitution:     L^T x = y
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param ch Precomputed CholeskyResult.
 * @param b  Right-hand side vector.
 *
 * @return Solution vector x.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
Vector<T, N> CholeskySolve(const CholeskyResult<T, N>& ch, Vector<T, N> b);

/**
 * @brief Convenience overload: decomposes A then solves Ax = b.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param A Square N×N SPD matrix.
 * @param b Right-hand side vector.
 *
 * @return Solution vector x.
 *
 * @throws std::runtime_error if @p A is not positive-definite.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
Vector<T, N> CholeskySolve(const Matrix<T, N, N>& A, Vector<T, N> b);

/**
 * @brief Computes det(A) from a precomputed Cholesky decomposition.
 *
 * \f[
 *   \det(A) = \det(L)^2 = \left(\prod_{i=0}^{N-1} L_{ii}\right)^2
 * \f]
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Matrix dimension.
 *
 * @param ch Precomputed CholeskyResult from CholeskyDecompose().
 *
 * @return Determinant of A.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
T CholeskyDeterminant(const CholeskyResult<T, N>& ch);

} // namespace lumina::linearalgebra

#include "../detail/linearalgebra/cholesky.tpp"
