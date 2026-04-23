namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Jacobi method: at each iteration every component of x is updated using only
// values from the previous iteration. The update rule for component i is:
//
//   x_new[i] = (b[i] - sum_{j != i} A(i,j) * x_old[j]) / A(i,i)
//
// Because x_new and x_old are kept separate, all components are updated in
// parallel (conceptually). This makes Jacobi easy to parallelise but converges
// more slowly than Gauss-Seidel.
//
// Convergence is guaranteed when A is strictly diagonally dominant.
// The step norm ||x_new - x_old|| is used as the stopping criterion.
template<typename T, std::size_t N>
SolverResult<T, N> Jacobi(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                          Vector<T, N> x0, int max_iter, T tol)
{
    Vector<T, N> x    = x0;
    Vector<T, N> x_new{};
    int          iter = 0;
    double       res  = 0.0;

    for (; iter < max_iter; ++iter)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            // Sum all off-diagonal contributions using the old x.
            T sigma = T{};
            for (std::size_t j = 0; j < N; ++j)
                if (j != i) sigma += A(i, j) * x[j];

            x_new[i] = (b[i] - sigma) / A(i, i);
        }

        // Measure the step size; stop when the update is below tolerance.
        res = static_cast<double>((x_new - x).Norm());
        x   = x_new;
        if (res < static_cast<double>(tol)) { ++iter; break; }
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

// Gauss-Seidel method: like Jacobi, but each component immediately uses the
// most recently computed values of x within the same iteration. The update
// for component i is:
//
//   x[i] = (b[i] - sum_{j < i} A(i,j)*x[j]  (already updated this iter)
//                 - sum_{j > i} A(i,j)*x[j]) (still from previous iter)
//           / A(i,i)
//
// In-place updating typically halves the number of iterations versus Jacobi
// for the same problem, at the cost of sequential data dependency.
//
// Convergence is guaranteed for strictly diagonally dominant or SPD matrices.
// The residual norm ||Ax - b|| is used as the stopping criterion.
template<typename T, std::size_t N>
SolverResult<T, N> GaussSeidel(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                               Vector<T, N> x, int max_iter, T tol)
{
    int    iter = 0;
    double res  = 0.0;

    for (; iter < max_iter; ++iter)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            // Use the already-updated x[j] for j < i and the old x[j] for j > i.
            T sigma = T{};
            for (std::size_t j = 0; j < N; ++j)
                if (j != i) sigma += A(i, j) * x[j];

            x[i] = (b[i] - sigma) / A(i, i);
        }

        // Compute the residual r = b - Ax and stop when ||r|| < tol.
        Vector<T, N> r = b - A * x;
        res = static_cast<double>(r.Norm());
        if (res < static_cast<double>(tol)) { ++iter; break; }
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

// Conjugate Gradient method: an iterative Krylov-subspace solver for SPD
// systems. It finds the exact solution (in exact arithmetic) in at most N
// steps, but in practice converges far sooner for well-conditioned systems.
//
// The algorithm maintains three vectors:
//   x  — current solution estimate
//   r  — residual r = b - Ax
//   p  — search direction (conjugate to all previous directions w.r.t. A)
//
// Each iteration:
//   1. Compute alpha = (r·r) / (p·Ap) — the step length along p.
//   2. Update x += alpha * p
//   3. Update r -= alpha * Ap
//   4. Compute beta = (r_new·r_new) / (r_old·r_old) — the direction update.
//   5. Update p = r_new + beta * p  — new search direction, A-conjugate to p.
//
// Convergence is measured against the relative residual ||r|| / ||b||.
template<typename T, std::size_t N>
SolverResult<T, N> ConjugateGradient(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                                     Vector<T, N> x, int max_iter, T tol)
{
    // Initial residual r = b - Ax, initial search direction p = r.
    Vector<T, N> r = b - A * x;
    Vector<T, N> p = r;

    T      r_dot    = r.Dot(r);               // Cached r·r, updated each iter.
    double b_norm   = static_cast<double>(b.Norm());
    int    iter     = 0;
    double res      = 0.0;

    // Guard against a zero right-hand side — solution is x = 0.
    if (b_norm < static_cast<double>(std::numeric_limits<T>::epsilon()))
        return { x, 0, 0.0, true };

    for (; iter < max_iter; ++iter)
    {
        Vector<T, N> Ap    = A * p;            // Matrix-vector product for this step.
        T            pAp   = p.Dot(Ap);        // Denominator of the step length.

        // If p·Ap is (near) zero the search direction is degenerate — A may not be SPD.
        if (std::abs(pAp) < std::numeric_limits<T>::epsilon())
            break;

        T alpha = r_dot / pAp;                 // Step length along p.

        x = x + p * alpha;                     // Advance solution.
        r = r - Ap * alpha;                    // Update residual.

        T r_dot_new = r.Dot(r);

        res = static_cast<double>(r.Norm()) / b_norm;
        if (res < static_cast<double>(tol)) { ++iter; break; }

        T beta = r_dot_new / r_dot;            // Direction update scalar.
        p      = r + p * beta;                 // New A-conjugate search direction.
        r_dot  = r_dot_new;
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

} // namespace lumina::linearalgebra
