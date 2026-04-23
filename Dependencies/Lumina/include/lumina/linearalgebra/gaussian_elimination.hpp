/**
 * @file gaussian_elimination.hpp
 * 
 * @brief Direct linear solvers based on Gaussian elimination.
 * 
 * @ingroup linearalgebra
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <limits>

#include <lumina/core/vector.hpp>
#include <lumina/core/matrix.hpp>

namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

/**
 * @brief Solves \f$A\mathbf{x} = \mathbf{b}\f$ using Gaussian elimination with partial pivoting.
 *
 * Reduces @p A to upper-triangular form via forward elimination, then recovers
 * \f$\mathbf{x}\f$ via back substitution:
 * \f[
 *   A\mathbf{x} = \mathbf{b}
 *   \;\Longrightarrow\;
 *   U\mathbf{x} = \mathbf{b}'
 * \f]
 *
 * @note Partial pivoting selects the row with the largest absolute value in each
 *       column as the pivot, improving numerical stability.
 *
 * @tparam T Scalar type.
 * @tparam N Matrix/vector dimensions.
 *
 * @param A Square N×N coefficient matrix.
 * @param b Right-hand side vector.
 *
 * @return Solution vector.
 *
 * @throws std::runtime_error if the matrix is singular or nearly singular.
 *
 * @ingroup linearalgebra
 *
 * @code
 * lumina::core::Matrix3d A{{4.0, 1.0, -1.0},
 *                          {2.0, 7.0,  1.0},
 *                          {1.0, 1.0,  3.0}};
 * lumina::core::Vec3d b{3.0, 19.0, 10.0};
 * auto x = lumina::linearalgebra::GaussianSolve(A, b);
 * @endcode
 */
template<typename T, std::size_t N>
Vector<T, N> GaussianSolve(Matrix<T, N, N> A, Vector<T, N> b);

/**
 * @brief Computes the inverse of matrix @p A.
 * 
 * @tparam T Scalar type.
 * @tparam N Matrix dimensions.
 * 
 * @param A Square NxN matrix.
 * 
 * @return Inverse NxN matrix.
 * 
 * @throws std::runtime_error if the matrix is singular.
 * 
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
Matrix<T, N, N> Inverse(Matrix<T, N, N> A);

/**
 * @brief Computes the determinant of @p A.
 * 
 * @tparam T Scalar type.
 * @tparam N Matrix dimensions.
 * 
 * @param A Square NxN matrix.
 * 
 * @return Scalar determinant.
 * 
 * @throws std::runtime_error if matrix is singular.
 * 
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
T Determinant(Matrix<T, N, N> A);

} // namespace lumina::linearalgebra

#include "../detail/linearalgebra/gaussian_elimination.tpp"