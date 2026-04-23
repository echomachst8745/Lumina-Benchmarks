/**
 * @file matrix.hpp
 * 
 * @brief Fixed size, stack allocated dense matrix class template stored in row major order.
 * 
 * @ingroup core
 */
#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <lumina/core/vector.hpp>

namespace lumina::core {

/**
 * @brief Fixed size, stack allocated dense matrix class template stored in row major order.
 *
 * @tparam T Scalar element type. Must be arithmetic (int, float, double, ...).
 * @tparam N Number of rows in matrix. Must be greater than 0.
 * @tparam M Number of columns in matrix. Must be greater than 0.
 *
 * @ingroup core
 *
 * @code
 * lumina::core::Matrix3d A{{1.0, 2.0, 3.0},
 *                          {4.0, 5.0, 6.0},
 *                          {7.0, 8.0, 9.0}};
 * std::cout << A.Trace() << "\n";       // 15.0
 * std::cout << A.Prettify() << "\n";    // aligned rows
 * auto At = A.Transpose();
 * @endcode
 */
template<typename T, std::size_t N, std::size_t M>
class Matrix
{
    static_assert(N > 0, "Matrix dimension must be greater than 0");
    static_assert(M > 0, "Matrix dimension must be greater than 0");
	static_assert(std::is_arithmetic_v<T>, "Matrix type must be arithmetic");

public:
    using value_type = T; ///< Element type.
    using size_type  = std::size_t; ///< Size type.

    static constexpr size_type NumRows = N; ///< Row count.
    static constexpr size_type NumCols = M; ///< Column count.

    // ----- Constructors -----

    /// @brief Initialises all matrix elements to zero.
    constexpr Matrix() noexcept;

    /**
     * @brief Fills matrix with @p fill_value.
     * 
     * @param fill_value Value to fill all elements with.
     */
    constexpr explicit Matrix(T fill_value) noexcept;

    /**
     * @brief Constructs matrix using a flat row major initialiser list.
     * 
     * @param init_list Initialiser list. Must contain exactly @p N * @p M values.
     * 
     * @throws std::invalid_argument if `init_list.size() != @p N * @p M `.
     */
    constexpr Matrix(std::initializer_list<T> init_list);

    /**
     * @brief Constructs matrix using a nested row by row initialiser list.
     * 
     * @param init_list_rows Nested initialiser list. Must contain exacly @p N rows and each row must contain @p M values.
     * 
     * @throws std::invalid_argument if `init_list_rows.size() != @p N ` and/or `init_list_rows[i].size() != @p M `.
     */
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> init_list_rows);

    // ----- Common Matrices -----

    /// @brief Returns a matrix with all zeros.
    static constexpr Matrix Zeros() noexcept;

    /// @brief Returns a matrix with all ones.
    static constexpr Matrix Ones() noexcept;

    /**
     * @brief Returns the NxN identity matrix.
     * 
     * @note Asserts NumRows == NumCols.
     */
    static constexpr Matrix Identity() noexcept;

    // ----- Dimension getters -----

    /// @brief Returns the number of rows in the matrix.
    static constexpr size_type Rows() noexcept;

    /// @brief Returns the number of columns in the matrix.
    static constexpr size_type Cols() noexcept;

    /// @brief Returns the total number of elements in the matrix.
    static constexpr size_type size() noexcept;

    /// @brief Returns true when NumRows == NumCols.
    static constexpr bool IsSquare() noexcept;

    // ----- Element access -----

    /**
     * @brief Bounds checked mutable element access.
     * 
     * @param i Zero based row index.
     * @param j Zero based column index.
     * 
     * @return Reference to the element.
     * 
     * @throws std::out_of_range if `i < 0' or 'j < 0` or 'i >= N' or 'j >= M'.
     */
    T& at(size_type i, size_type j);

    /**
     * @brief Bounds checked const element access.
     * 
     * @param i Zero based row index.
     * @param j Zero based column index.
     * 
     * @return Const reference to the element.
     * 
     * @throws std::out_of_range if `i < 0' or 'j < 0` or 'i >= N' or 'j >= M'.
     */
    const T& at(size_type i, size_type j) const;

    /**
     * @brief Unchecked mutable element access.
     * 
     * @param i Zero based row index.
     * @param j Zero based column index.
     * 
     * @return Reference to the element.
     */
    constexpr T& operator()(size_type i, size_type j) noexcept;

    /**
     * @brief Unchecked const element access.
     * 
     * @param i Zero based row index.
     * @param j Zero based column index.
     * 
     * @return Const reference to the element.
     */
    constexpr const T& operator()(size_type i, size_type j) const noexcept;

    /// @brief Returns a pointer to the underlying data array.
    constexpr T* data() noexcept;

    /// @brief Returns a const pointer to the underlying data array.
    constexpr const T* data() const noexcept;

    // ----- Row / column access -----

    /// @brief Returns a copy of row at index @p i .
    std::array<T, M> Row(size_type i) const;

    /// @brief Returns a copy of column at index @p i .
    std::array<T, N> Col(size_type i) const;

    /**
     * @brief Sets row at index @p i with values in @p row_values .
     * 
     * @param i Zero based row index.
     * @param row_values Values to set in row. 
     */
    void SetRow(size_type i, const std::array<T, M>& row_values);

    /**
     * @brief Sets column at index @p i with values in @p col_values .
     *
     * @param i Zero based column index.
     * @param col_values Values to set in column.
     */
    void SetCol(size_type i, const std::array<T, N>& col_values);

    /**
     * @brief Swaps rows at indices @p i and @p j in place.
     *
     * @param i Zero based index of the first row.
     * @param j Zero based index of the second row.
     */
    void SwapRows(size_type i, size_type j) noexcept;

    /**
     * @brief Swaps columns at indices @p i and @p j in place.
     *
     * @param i Zero based index of the first column.
     * @param j Zero based index of the second column.
     */
    void SwapCols(size_type i, size_type j) noexcept;

    // ----- Arithmetic operations -----
    // [[nodiscard]] on all arithmetic methods: each produces a new Matrix and does not
    // modify *this — silently discarding the return value is almost certainly a bug.

    /**
     * @brief Matrix addition function.
     *
     * @param other Other Matrix.
     *
     * @return A new matrix with addition operation performed.
     */
    [[nodiscard]] constexpr Matrix Add(const Matrix& other) const noexcept;

    /**
     * @brief Matrix addition operator.
     *
     * @param other Other Matrix.
     *
     * @return New matrix with addition operation performed.
     */
    [[nodiscard]] constexpr Matrix operator+(const Matrix& other) const noexcept;

    /**
     * @brief Matrix subtraction function.
     *
     * @param rhs Right hand side matrix.
     *
     * @return New matrix with subtraction operation performed.
     */
    [[nodiscard]] constexpr Matrix Subtract(const Matrix& rhs) const noexcept;

    /**
     * @brief Matrix subtraction operator.
     *
     * @param rhs Right hand side matrix.
     *
     * @return New matrix with subtraction operation performed.
     */
    [[nodiscard]] constexpr Matrix operator-(const Matrix& rhs) const noexcept;

    /**
     * @brief Matrix scalar multiplication function.
     *
     * @param scalar Scalar muliplier.
     *
     * @return New Matrix with scalar multiplication operation performed.
     */
    [[nodiscard]] constexpr Matrix Multiply(T scalar) const noexcept;

    /**
     * @brief Matrix scalar multiplication operator.
     *
     * @param scalar Scalar muliplier.
     *
     * @return New Matrix with scalar multiplication operation performed.
     */
    [[nodiscard]] constexpr Matrix operator*(T scalar) const noexcept;

    /**
     * @brief Matrix scalar division function.
     *
     * @param scalar Scalar divisor.
     *
     * @return New Matrix with scalar division operation performed.
     */
    [[nodiscard]] constexpr Matrix Divide(T scalar) const noexcept;

    /**
     * @brief Matrix scalar division operator.
     *
     * @param scalar Scalar divisor.
     *
     * @return New Matrix with scalar division operation performed.
     */
    [[nodiscard]] constexpr Matrix operator/(T scalar) const noexcept;

    /**
     * @brief In place matrix addition operator.
     * 
     * @param other Other matrix.
     * 
     * @return Self reference.
     */
    constexpr Matrix& operator+=(const Matrix& other) noexcept;

    /**
     * @brief In place matrix subtraction operator.
     * 
     * @param other Right hand side matrix.
     * 
     * @return Self reference.
     */
    constexpr Matrix& operator-=(const Matrix& rhs) noexcept;

    /**
     * @brief In place matrix multiplication operator.
     * 
     * @param scalar Scalar multiplier.
     * 
     * @return Self reference.
     */
    constexpr Matrix& operator*=(T scalar) noexcept;

    /**
     * @brief In place matrix division operator.
     * 
     * @param scalar Scalar divisor.
     * 
     * @return Self reference.
     */
    constexpr Matrix& operator/=(T scalar) noexcept;

    /**
     * @brief Matrix on matrix multiplication function.
     *
     * @tparam OtherCols Number of column in @p rhs.
     *
     * @param rhs Right hand side matrix.
     *
     * @return New matrix with multiplication operation performed.
     */
    template<std::size_t OtherCols>
    [[nodiscard]] constexpr Matrix<T, N, OtherCols> Multiply(const Matrix<T, M, OtherCols>& rhs) const noexcept;

    /**
     * @brief Matrix on matrix multiplication operator.
     *
     * @tparam OtherCols Number of column in @p rhs.
     *
     * @param rhs Right hand side matrix.
     *
     * @return New matrix with multiplication operation performed.
     */
    template<std::size_t OtherCols>
    [[nodiscard]] constexpr Matrix<T, N, OtherCols> operator*(const Matrix<T, M, OtherCols>& rhs) const noexcept;

    /**
     * @brief Matrix on vector multiplication function.
     *
     * @tparam VectorSize Number of elements in @p vector . Must equal @p N.
     *
     * @param vector Vector to multiply with.
     *
     * @return New vector with multiplication applied.
     */
    template<std::size_t VectorSize>
    [[nodiscard]] constexpr Vector<T, N> Multiply(const Vector<T, VectorSize>& vector) const;

    /**
     * @brief Matrix on vector multiplication operator.
     *
     * @tparam VectorSize Number of elements in @p vector . Must equal @p N.
     *
     * @param vector Vector to multiply with.
     *
     * @return New vector with multiplication applied.
     */
    template<std::size_t VectorSize>
    [[nodiscard]] constexpr Vector<T, N> operator*(const Vector<T, VectorSize>& vector) const;

    // ----- Matrix operations -----
    // [[nodiscard]] on all matrix operations: each computes and returns a new value;
    // silently discarding the result is almost certainly a bug.

    /**
     * @brief Matrix transpose function: \f$B_{ij} = A_{ji}\f$.
     *
     * @return New matrix with transpose operation applied.
     */
    [[nodiscard]] constexpr Matrix<T, M, N> Transpose() const noexcept;

    /**
     * @brief Computes the Frobenius norm of the matrix: \f$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}\f$.
     *
     * @return Frobenius norm as type @p T.
     */
    [[nodiscard]] T FrobeniusNorm() const noexcept;

    /**
     * @brief Computes the sum of the diagonal elements: \f$\operatorname{tr}(A) = \sum_{i} A_{ii}\f$.
     *
     * @note Matrix must be square.
     *
     * @return Diagonal sum value as type @p T.
     */
    [[nodiscard]] T Trace() const;

    // ----- Matrix properties -----

    /**
     * @brief Returns true when \f$A = A^T\f$ (within tolerance @p delta).
     *
     * @param delta Per-element absolute tolerance for the symmetry check.
     */
    [[nodiscard]] bool IsSymmetric(T delta = T{1e-10}) const;

    // ------ Matrix comparison -----

    /**
     * @brief Element equality comparison function.
     * 
     * @param other Other matrix.
     * 
     * @return True if elements are equal.
     */
    constexpr bool Equals(const Matrix& other) const noexcept;

    /**
     * @brief Element equality comparison operator.
     * 
     * @param other Other matrix.
     * 
     * @return True if elements are equal.
     */
    constexpr bool operator==(const Matrix& other) const noexcept;

    /**
     * @brief Element inequality comparison operator.
     *
     * @param other Other matrix.
     *
     * @return True if elements are not equal.
     */
    constexpr bool operator!=(const Matrix& other) const noexcept;

    // ----- String conversion -----

    /**
     * @brief Returns a string representation of the matrix.
     *
     * @return String in the format `[[a, b], [c, d]]`.
     */
    std::string ToString() const;

    /**
     * @brief Returns a multi-line, column-aligned string representation of the matrix.
     *
     * @return Multi-line aligned string, e.g. for a 2×3 matrix:
     * @code
     * [  1,  2,  3 ]
     * [  4,  5,  6 ]
     * @endcode
     */
    std::string Prettify() const;

    /// @brief Stream insertion operator.
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat)
    {
        return os << mat.ToString();
    }

private:
    // _simd_align: largest SIMD alignment that fits the object's actual byte size.
    //   N*M*sizeof(T) >= 32 → 32: AVX2 256-bit aligned loads/stores (e.g. Matrix4f = 64 B)
    //   N*M*sizeof(T) >= 16 → 16: SSE 128-bit aligned loads/stores (e.g. Matrix2d = 32 B)
    //   N*M*sizeof(T) <  16 → alignof(T): natural alignment, no padding waste
    // Without this, std::array only guarantees alignof(T), which prevents the compiler
    // from emitting aligned vector loads and suppresses auto-vectorisation.
    static constexpr std::size_t _simd_align =
        (N * M * sizeof(T) >= 32) ? 32u :
        (N * M * sizeof(T) >= 16) ? 16u :
        alignof(T);

    alignas(_simd_align) std::array<T, N * M> _data; ///< Fixed size array to store elements.
};

// ----- Aliases -----
// These are commonly used matrix types predefined for convenience.

template<typename T, std::size_t N>
using SquareMatrix = Matrix<T, N, N>; ///< Alias for an NxN matrix.

using Matrix2f = Matrix<float, 2, 2>; ///< Alias for a 2x2 float matrix.
using Matrix3f = Matrix<float, 3, 3>; ///< Alias for a 3x3 float matrix.
using Matrix4f = Matrix<float, 4, 4>; ///< Alias for a 4x4 float matrix.
using Matrix2d = Matrix<double, 2, 2>; ///< Alias for a 2x2 double matrix.
using Matrix3d = Matrix<double, 3, 3>; ///< Alias for a 3x3 double matrix.
using Matrix4d = Matrix<double, 4, 4>; ///< Alias for a 4x4 double matrix.

} // namespace lumina::core

#include "../detail/core/matrix.tpp"