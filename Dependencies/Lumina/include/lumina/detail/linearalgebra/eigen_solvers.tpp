namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Power Iteration to find the dominant eigenvalue/eigenvector of A.
//
// Algorithm:
//   1. Normalise the initial guess x.
//   2. Each iteration: q = Ax, then x = q / ||q||.
//   3. Compute the Rayleigh quotient λ = x^T A x as the eigenvalue estimate.
//   4. Converge when |λ_new - λ_old| < tol.
//
// If x0 is the zero vector, a uniform all-ones start is used instead so the
// caller does not need to supply a non-trivial initial guess.
template<typename T, std::size_t N>
EigenResult<T, N> PowerIteration(
    const Matrix<T, N, N>& A,
    Vector<T, N>           x0,
    std::size_t            max_iter,
    T                      tol)
{
    // Use uniform start if caller passed the zero vector.
    Vector<T, N> x = x0;
    if (x.NormSquared() < std::numeric_limits<T>::epsilon())
    {
        for (std::size_t i = 0; i < N; ++i)
            x[i] = T{1};
    }

    // Normalise initial vector.
    x = x / x.Norm();

    T           lambda_prev = std::numeric_limits<T>::max();
    std::size_t iter        = 0;
    bool        converged   = false;

    for (; iter < max_iter; ++iter)
    {
        Vector<T, N> q = A * x;

        T q_norm = q.Norm();
        if (q_norm < std::numeric_limits<T>::epsilon())
            break;  // Ax = 0: trivial null eigenvector, eigenvalue = 0.

        x = q / q_norm;

        // Rayleigh quotient: λ = x^T A x  (x is already unit-length).
        T lambda = (A * x).Dot(x);

        if (std::abs(lambda - lambda_prev) < tol)
        {
            converged = true;
            lambda_prev = lambda;
            ++iter;
            break;
        }
        lambda_prev = lambda;
    }

    return EigenResult<T, N>{ lambda_prev, x, iter, converged };
}

} // namespace lumina::linearalgebra
