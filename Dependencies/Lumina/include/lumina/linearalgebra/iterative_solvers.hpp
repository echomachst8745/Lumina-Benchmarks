/**
 * @file iterative_solvers.hpp
 *
 * @brief Iterative methods for solving fixed-size linear systems Ax = b.
 *
 * @ingroup iterativesolvers
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
 * @brief Result returned by all iterative solvers.
 * 
 * @tparam T Scalar type.
 * @tparam N System dimension.
 * 
 * @ingroup iterativesolvers
 */
template<typename T, std::size_t N>
struct SolverResult
{
    Vector<T, N> x;          ///< Approximate solution.
    int          iterations; ///< Iterations performed.
    double       residual;   ///< Final residual at termination.
    bool         converged;  ///< True if the solver met the tolerance.
};

/**
 * @brief Solves Ax = b via the Jacobi iterative method.
 *
 * Update rule per iteration:
 * \f[
 *   x_i^{(k+1)} = \frac{1}{A_{ii}}
 *     \left(b_i - \sum_{j \neq i} A_{ij}\, x_j^{(k)}\right)
 * \f]
 *
 * @note Convergence is guaranteed when @p A is strictly diagonally dominant.
 *       Stopping criterion: \f$\|x^{(k+1)} - x^{(k)}\| < \text{tol}\f$.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param A Square N×N coefficient matrix.
 * @param b Right-hand side vector.
 * @param x0 Initial guess (zero-initialised by default).
 * @param max_iter Maximum number of iterations.
 * @param tol Convergence tolerance on the step norm.
 *
 * @return SolverResult<T, N>.
 *
 * @ingroup iterativesolvers
 */
template<typename T, std::size_t N>
SolverResult<T, N> Jacobi(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                          Vector<T, N> x0 = {}, int max_iter = 1000, T tol = T{1e-10});

/**
 * @brief Solves Ax = b via the Gauss–Seidel iterative method.
 *
 * Uses the most recently computed components within each sweep:
 * \f[
 *   x_i^{(k+1)} = \frac{1}{A_{ii}}
 *     \left(b_i
 *       - \sum_{j < i} A_{ij}\, x_j^{(k+1)}
 *       - \sum_{j > i} A_{ij}\, x_j^{(k)}
 *     \right)
 * \f]
 *
 * @note Convergence guaranteed for strictly diagonally dominant or SPD matrices.
 *       Stopping criterion: \f$\|b - Ax\| < \text{tol}\f$.
 *       Typically converges in roughly half the iterations of Jacobi.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param A Square N×N coefficient matrix.
 * @param b Right-hand side vector.
 * @param x Initial guess (zero-initialised by default).
 * @param max_iter Maximum number of iterations.
 * @param tol Convergence tolerance on the residual norm.
 *
 * @return SolverResult<T, N>.
 *
 * @ingroup iterativesolvers
 */
template<typename T, std::size_t N>
SolverResult<T, N> GaussSeidel(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                               Vector<T, N> x = {}, int max_iter = 1000, T tol = T{1e-10});

/**
 * @brief Solves Ax = b via the Conjugate Gradient method (requires SPD A).
 *
 * Krylov-subspace method that converges in at most N steps in exact arithmetic.
 * Step length and direction update scalars:
 * \f[
 *   \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k},
 *   \qquad
 *   \beta_k  = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}
 * \f]
 *
 * @note @p A must be symmetric positive-definite (SPD).
 *       Stopping criterion: relative residual \f$\|r\|/\|b\| < \text{tol}\f$.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N System dimension.
 *
 * @param A SPD N×N coefficient matrix.
 * @param b Right-hand side vector.
 * @param x Initial guess (zero-initialised by default).
 * @param max_iter Maximum number of iterations.
 * @param tol Relative residual tolerance \f$\|r\|/\|b\|\f$.
 *
 * @return SolverResult<T, N>.
 *
 * @ingroup iterativesolvers
 *
 * @code
 * lumina::core::Matrix<double, 5, 5> A = make_spd<5>();
 * lumina::core::Vector<double, 5> b(1.0);
 * auto result = lumina::linearalgebra::ConjugateGradient(A, b);
 * if (result.converged)
 *     std::cout << result.x << "\n";
 * @endcode
 */
template<typename T, std::size_t N>
SolverResult<T, N> ConjugateGradient(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                                     Vector<T, N> x = {}, int max_iter = 1000, T tol = T{1e-10});

} // namespace lumina::linearalgebra

#include "../detail/linearalgebra/iterative_solvers.tpp"
