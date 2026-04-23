/**
 * @file eigen_solvers.hpp
 *
 * @brief Eigenvalue and eigenvector solvers for square matrices.
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
 * @brief Result of a Power Iteration solve.
 *
 * @tparam T Scalar element type.
 * @tparam N Matrix dimension.
 *
 * @ingroup linearalgebra
 */
template<typename T, std::size_t N>
struct EigenResult
{
    T            eigenvalue;  ///< Dominant eigenvalue (largest absolute value).
    Vector<T, N> eigenvector; ///< Corresponding unit eigenvector.
    std::size_t  iterations;  ///< Iterations performed.
    bool         converged;   ///< True if the tolerance was met before max_iter.
};

/**
 * @brief Finds the dominant eigenvalue/eigenvector of A via Power Iteration.
 *
 * Repeatedly computes \f$x_{k+1} = A x_k / \|A x_k\|\f$ until the Rayleigh
 * quotient \f$\lambda = x^T A x / x^T x\f$ converges.
 *
 * Convergence requires A to have a unique dominant eigenvalue (one whose
 * absolute value strictly exceeds all others). Works for symmetric and
 * non-symmetric matrices alike.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Matrix dimension.
 *
 * @param A        Square N×N matrix.
 * @param x0       Initial guess vector. If the zero vector is passed a
 *                 uniform [1, 1, ..., 1] vector is used instead.
 * @param max_iter Maximum number of iterations.
 * @param tol      Convergence tolerance on the absolute change in the
 *                 Rayleigh quotient between successive iterations.
 *
 * @return EigenResult containing the dominant eigenvalue, its unit
 *         eigenvector, iteration count, and convergence flag.
 *
 * @ingroup linearalgebra
 *
 * @code
 * lumina::core::Matrix3d A{{2.0, 1.0, 0.0},
 *                          {1.0, 3.0, 1.0},
 *                          {0.0, 1.0, 2.0}};
 * auto result = lumina::linearalgebra::PowerIteration(A);
 * // result.eigenvalue  ≈ 4.414 (dominant eigenvalue)
 * // result.eigenvector ≈ corresponding unit eigenvector
 * @endcode
 */
template<typename T, std::size_t N>
EigenResult<T, N> PowerIteration(
    const Matrix<T, N, N>& A,
    Vector<T, N>           x0       = Vector<T, N>{},
    std::size_t            max_iter = 1000,
    T                      tol      = T{1e-10});

} // namespace lumina::linearalgebra

#include "../detail/linearalgebra/eigen_solvers.tpp"
