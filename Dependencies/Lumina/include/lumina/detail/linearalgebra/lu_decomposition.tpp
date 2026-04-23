namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Computes the LU decomposition PA = LU with partial pivoting.
//
// The result stores three things:
//   L    — unit lower-triangular matrix (diagonal is all 1s)
//   U    — upper-triangular matrix
//   perm — the row permutation applied during pivoting (P encodes which
//           original row ended up in each position)
//   sign — parity of the permutation: +1 if an even number of swaps,
//          -1 if odd. Used by LUDeterminant to get the correct sign.
//
// Partial pivoting swaps the row with the largest absolute value in the
// current column to the diagonal position before elimination. This keeps
// the multipliers |L(i,j)| <= 1, preventing catastrophic error growth.
template<typename T, std::size_t N>
LUResult<T, N> LUDecompose(Matrix<T, N, N> A)
{
    LUResult<T, N> result;

    // Initialise perm as the identity permutation [0, 1, 2, ..., N-1].
    std::iota(result.perm.begin(), result.perm.end(), 0);
    result.sign = 1;

    for (std::size_t col = 0; col < N; ++col)
    {
        // Find the row at or below `col` with the largest absolute value —
        // the partial pivot.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular");

        if (pivot != col)
        {
            // Swap rows in A and record the swap in perm.
            // Each swap flips the sign of the determinant.
            A.SwapRows(col, pivot);
            std::swap(result.perm[col], result.perm[pivot]);
            result.sign = -result.sign;
        }

        // Compute multipliers for rows below the diagonal and eliminate.
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / A(col, col);
            A(row, col) = factor;               // Store multiplier in-place (lower triangle of A).
            for (std::size_t k = col + 1; k < N; ++k)
                A(row, k) -= factor * A(col, k);
        }
    }

    // Unpack the combined matrix into separate L and U.
    // L has 1s on the diagonal; the multipliers are stored below it.
    // U is the upper triangle including the diagonal.
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            if (i > j)
            {
                result.L(i, j) = A(i, j);   // Multiplier stored below diagonal.
                result.U(i, j) = T{};
            }
            else if (i == j)
            {
                result.L(i, j) = T{1};       // Unit diagonal of L.
                result.U(i, j) = A(i, j);
            }
            else
            {
                result.L(i, j) = T{};
                result.U(i, j) = A(i, j);
            }
        }
    }

    return result;
}

// Solves Ax = b given a precomputed LU decomposition of A (PA = LU).
//
// Two triangular solves are performed:
//   1. Forward substitution:  Ly = Pb  (L is unit lower-triangular)
//   2. Back substitution:     Ux = y   (U is upper-triangular)
//
// Applying the permutation P first reorders b to match the row swaps
// that were recorded during decomposition.
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const LUResult<T, N>& lu, Vector<T, N> b)
{
    // Apply the row permutation P to b: pb[i] = b[perm[i]].
    Vector<T, N> pb{};
    for (std::size_t i = 0; i < N; ++i)
        pb[i] = b[lu.perm[i]];

    // Forward substitution: solve Ly = pb.
    // L has 1s on the diagonal so no division is needed.
    Vector<T, N> y{};
    for (std::size_t i = 0; i < N; ++i)
    {
        T sum = pb[i];
        for (std::size_t j = 0; j < i; ++j)
            sum -= lu.L(i, j) * y[j];
        y[i] = sum;
    }

    // Back substitution: solve Ux = y.
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        T sum = y[i];
        for (std::size_t j = i + 1; j < N; ++j)
            sum -= lu.U(i, j) * x[j];
        x[i] = sum / lu.U(i, i);
    }

    return x;
}

// Convenience overload: decomposes A on the fly then solves.
// Use the two-argument form when solving multiple right-hand sides
// against the same A to avoid recomputing the decomposition each time.
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const Matrix<T, N, N>& A, Vector<T, N> b)
{
    return LUSolve(LUDecompose(A), b);
}

// Computes det(A) from a precomputed LU decomposition.
//
// det(PA) = det(L) * det(U). Since L has unit diagonal, det(L) = 1.
// det(U) is the product of its diagonal entries.
// det(P) = sign (±1 depending on the parity of row swaps).
// Therefore: det(A) = sign * Π U(i,i).
template<typename T, std::size_t N>
T LUDeterminant(const LUResult<T, N>& lu)
{
    T det = static_cast<T>(lu.sign);
    for (std::size_t i = 0; i < N; ++i)
        det *= lu.U(i, i);
    return det;
}

} // namespace lumina::linearalgebra
