namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Solves Ax = b using Gaussian elimination with partial pivoting.
//
// A and b are taken by value so the originals are not modified.
//
// The algorithm has two phases:
//   1. Forward elimination — reduce A to upper triangular form.
//   2. Back substitution  — solve the triangular system for x.
//
// Partial pivoting reorders rows at each step to place the largest absolute
// value in the current column on the diagonal. This prevents division by small
// numbers that would amplify floating-point errors.
template<typename T, std::size_t N>
Vector<T, N> GaussianSolve(Matrix<T, N, N> A, Vector<T, N> b)
{
    // ----- Phase 1: forward elimination -----
    for (std::size_t col = 0; col < N; ++col)
    {
        // Find the row at or below `col` with the largest absolute value in
        // this column — the partial pivot.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        // If the largest value is (near) zero the column is linearly dependent
        // and the system has no unique solution.
        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular or nearly singular");

        // Swap the pivot row into position. The same swap is applied to b so
        // the augmented system [A|b] stays consistent.
        if (pivot != col) { A.SwapRows(col, pivot); std::swap(b[col], b[pivot]); }

        T diag = A(col, col);

        // Eliminate all entries below the diagonal in this column.
        // For each row below `col`, compute the multiplier and subtract the
        // scaled pivot row so the entry at (row, col) becomes zero.
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / diag;
            if (factor == T{}) continue;                 // Already zero, skip.
            for (std::size_t k = col; k < N; ++k)
                A(row, k) -= factor * A(col, k);
            b[row] -= factor * b[col];                   // Mirror the operation on b.
        }
    }

    // ----- Phase 2: back substitution -----
    // A is now upper triangular. Solve from the last row upward:
    // x[i] = (b[i] - sum of A(i,j)*x[j] for j > i) / A(i,i)
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        T sum = b[i];
        for (std::size_t j = i + 1; j < N; ++j)
            sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }

    return x;
}

// Computes A^{-1} using Gauss-Jordan elimination on the augmented matrix [A | I].
//
// The same row operations that reduce A to the identity are applied to I in
// parallel. When A becomes I, what started as I has become A^{-1}.
//
// Unlike GaussianSolve, this uses full elimination (above and below the
// diagonal) plus row normalisation, so no back substitution step is needed.
template<typename T, std::size_t N>
Matrix<T, N, N> Inverse(Matrix<T, N, N> A)
{
    // Start with the NxN identity; row operations will transform it into A^{-1}.
    Matrix<T, N, N> inv = Matrix<T, N, N>::Identity();

    for (std::size_t col = 0; col < N; ++col)
    {
        // Partial pivot: find the row with the largest value in this column.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular");

        // Apply the same row swap to both A and inv to keep them in sync.
        if (pivot != col) { A.SwapRows(col, pivot); inv.SwapRows(col, pivot); }

        T diag = A(col, col);

        // Normalise the pivot row so the diagonal entry becomes 1.
        // This is the "Jordan" part — it allows full elimination without a
        // separate back substitution phase.
        for (std::size_t k = 0; k < N; ++k) { A(col,k)/=diag; inv(col,k)/=diag; }

        // Eliminate all other entries in this column (above and below),
        // keeping inv synchronised with every operation applied to A.
        for (std::size_t row = 0; row < N; ++row)
        {
            if (row == col) continue;               // Skip the pivot row itself.
            T factor = A(row, col);
            if (factor == T{}) continue;            // Entry is already zero.
            for (std::size_t k = 0; k < N; ++k)
            {
                A(row,k)   -= factor * A(col,k);
                inv(row,k) -= factor * inv(col,k);
            }
        }
    }

    return inv;
}

// Computes det(A) using partial-pivoting Gaussian elimination.
//
// The determinant of an upper triangular matrix is the product of its diagonal
// entries. Gaussian elimination reduces A to upper triangular form, so we just
// accumulate the diagonal product as we go.
//
// Row swaps flip the sign of the determinant (each swap multiplies det by -1),
// so a sign variable tracks the net parity.
template<typename T, std::size_t N>
T Determinant(Matrix<T, N, N> A)
{
    T det = T{1};

    for (std::size_t col = 0; col < N; ++col)
    {
        // Partial pivot.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        // Singular matrix — determinant is zero by definition.
        if (max_val < std::numeric_limits<T>::epsilon() * 1000) return T{};

        // Each row swap negates the determinant.
        if (pivot != col) { A.SwapRows(col, pivot); det = -det; }

        // Accumulate the diagonal entry into the determinant product.
        det *= A(col, col);
        T diag = A(col, col);

        // Eliminate entries below the diagonal (same as GaussianSolve, but
        // we don't need to carry a right-hand side here).
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / diag;
            for (std::size_t k = col; k < N; ++k)
                A(row,k) -= factor * A(col,k);
        }
    }

    return det;
}

} // namespace lumina::linearalgebra
