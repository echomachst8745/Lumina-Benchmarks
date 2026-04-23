/**
* @file vector.hpp
* 
* @brief Fixed size, stack allocated vector class template.
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

namespace lumina::core {

/**
* @brief Fixed size, stack allocated mathematical vector.
*
* @tparam T Scalar element type. Must be arithmetic (int, float, double, ...).
* @tparam N Number of elements. Must be greater than 0.
*
* @ingroup core
*
* @code
* lumina::core::Vec3d v{1.0, 2.0, 3.0};
* auto n = v.Normalised();
* double d = v.Dot(n);            // ~3.742
* std::cout << v << "\n";         // [1, 2, 3]
* std::cout << v.Prettify();      // column format
* @endcode
*/
template<typename T, std::size_t N>
class Vector
{
	static_assert(N > 0, "Vector size must be greater than 0");
	static_assert(std::is_arithmetic_v<T>, "Vector type must be arithmetic");

public:
	using value_type     = T; ///< Element type.
	using size_type      = std::size_t; ///< Size type.
	using iterator       = typename std::array<T, N>::iterator; ///< Mutable iterator.
	using const_iterator = typename std::array<T, N>::const_iterator; ///< Immutable iterator.

	// ----- Constructors ------

	/// @brief Initialises all vector elements to zero.
	constexpr Vector() noexcept;

	/**
	* @brief Fills vector with @p fill_value .
	* 
	* @param fill_value Value to fill all elements with.
	*/
	constexpr explicit Vector(T fill_value) noexcept;

	/**
	* @brief Initializes vector elements using a braced initialiser list.
	* 
	* @param init_list Initialiser list. Must contain exactly @p N values.
	* 
	* @throws std::invalid_argument if `init_list.size() != @p N `.
	*/
	constexpr Vector(std::initializer_list<T> init_list);

	// ------ Element Access -----

	/**
	* @brief Bounds checked mutable element access.
	* 
	* @param i Zero based index.
	* 
	* @return Reference to the element.
	* 
	* @throws std::out_of_range if `i < 0` or `i >= N`.
	*/
	T& at(size_type i);

	/**
	* @brief Bounds checked const element access.
	*
	* @param i Zero based index.
	*
	* @return Const reference to the element.
	*
	* @throws std::out_of_range if `i < 0` or `i >= N`.
	*/
	const T& at(size_type i) const;

	/**
	* @brief Unchecked mutable element access.
	* 
	* @param i Zero based index.
	* 
	* @return Reference to the element.
	*/
	constexpr T& operator[](size_type i) noexcept;

	/**
	* @brief Unchecked const element access.
	*
	* @param i Zero based index.
	*
	* @return Const reference to the element.
	*/
	constexpr const T& operator[](size_type i) const noexcept;

	/// @brief Returns a pointer to the underlying data array.
	constexpr T* data() noexcept;

	/// @brief Returns a const pointer to the underlying data array.
	constexpr const T* data() const noexcept;
	
	/// @brief Returns the number of elements @p N.
	constexpr size_type size() const noexcept;

	// ----- Iterators -----

	constexpr iterator       begin()  noexcept;  ///< Begin iterator.
	constexpr iterator       end()    noexcept;  ///< End iterator.

	constexpr const_iterator begin()  const noexcept; ///< Const begin.
	constexpr const_iterator end()    const noexcept; ///< Const end.
	constexpr const_iterator cbegin() const noexcept; ///< cbegin.
	constexpr const_iterator cend()   const noexcept; ///< cend.

	// ----- Arithmetic operations -----
	// [[nodiscard]] on all arithmetic methods: each produces a new Vector and does not
	// modify *this — silently discarding the return value is almost certainly a bug.

	/**
	* @brief Vector addition function.
	*
	* @param other Other vector.
	*
	* @return New vector with addition operation performed.
	*/
	[[nodiscard]] constexpr Vector Add(const Vector& other) const noexcept;

	/**
	* @brief Vector addition operator.
	*
	* @param other Other vector.
	*
	* @return New vector with addition operation performed.
	*/
	[[nodiscard]] constexpr Vector operator+(const Vector& other) const noexcept;

	/**
	* @brief Vector subtraction function.
	*
	* @param rhs Right hand side vector.
	*
	* @return New vector with subtraction operation performed.
	*/
	[[nodiscard]] constexpr Vector Subtract(const Vector& rhs) const noexcept;

	/**
	* @brief Vector subtraction operator.
	*
	* @param rhs Right hand side vector.
	*
	* @return New vector with subtraction operation performed.
	*/
	[[nodiscard]] constexpr Vector operator-(const Vector& rhs) const noexcept;

	/**
	* @brief Vector negation operator.
	*
	* @return New vector with negation operation performed.
	*/
	[[nodiscard]] constexpr Vector operator-() const noexcept;

	/**
	* @brief Vector scalar multiplication function.
	*
	* @param scalar Scalar multiplier.
	*
	* @return New vector with scalar multiplication operation performed.
	*/
	[[nodiscard]] constexpr Vector Multiply(T scalar) const noexcept;

	/**
	* @brief Vector scalar multiplication operator.
	*
	* @param scalar Scalar multiplier.
	*
	* @return New vector with scalar multiplication operation performed.
	*/
	[[nodiscard]] constexpr Vector operator*(T scalar) const noexcept;

	/**
	* @brief Vector scalar division function.
	*
	* @param scalar Scalar divisor.
	*
	* @return New vector with scalar division operation performed.
	*/
	[[nodiscard]] constexpr Vector Divide(T scaler) const noexcept;

	/**
	* @brief Vector scalar division operator.
	*
	* @param scalar Scalar divisor.
	*
	* @return New vector with scalar division operation performed.
	*/
	[[nodiscard]] constexpr Vector operator/(T scalar) const noexcept;

	/**
	* @brief In place Vector addition operator.
	*
	* @param other Other vector.
	*
	* @return Self reference.
	*/
	constexpr Vector& operator+=(const Vector& other) noexcept;
	
	/**
	* @brief In place Vector subtraction operator.
	*
	* @param rhs Right hand side vector.
	*
	* @return Self reference.
	*/
	constexpr Vector& operator-=(const Vector& rhs) noexcept;
	
	/**
	* @brief In place vector scalar multiplication operator.
	*
	* @param scalar Scalar multiplier.
	*
	* @return Self reference.
	*/
	constexpr Vector& operator*=(T scaler) noexcept;
	
	/**
	* @brief In place vector scalar division operator.
	*
	* @param scalar Scalar divisor.
	*
	* @return Self reference.
	*/
	constexpr Vector& operator/=(T scaler) noexcept;

	// ----- Comparison operations -----

	/**
	* @brief Element equality comparision function.
	* 
	* @param other Other vector.
	* 
	* @return True if elements are equal.
	*/
	constexpr bool Equals(const Vector& other) const noexcept;

	/**
	* @brief Element equality comparision operator.
	*
	* @param other Other vector.
	*
	* @return True if elements are equal.
	*/
	constexpr bool operator==(const Vector& other) const noexcept;

	/**
	* @brief Element inequality comparision operator.
	*
	* @param other Other vector.
	*
	* @return True if elements are not equal.
	*/
	constexpr bool operator!=(const Vector& other) const noexcept;

	// ----- Mathematical operations -----
	// [[nodiscard]] on all mathematical methods: each computes and returns a new value;
	// silently discarding the result is almost certainly a bug.

	/**
	* @brief Computes the dot product with another vector: \f$\sum_{i=0}^{N-1} a_i\, b_i\f$.
	*
	* @param rhs Right hand side vector.
	*
	* @return Dot product value as type @p T.
	*/
	[[nodiscard]] constexpr T Dot(const Vector& rhs) const noexcept;

	/**
	* @brief Computes the Euclidean norm: \f$\sqrt{\sum{{x_i}^2}}\f$.
	*
	* @return Euclidean norm value as type @p T.
	*/
	[[nodiscard]] T Norm() const noexcept;

	/**
	* @brief Computes the squared Euclidean norm: \f$\sum{{x_i}^2}\f$.
	*
	* @return Squared euclidean norm value as type @p T.
	*/
	[[nodiscard]] T NormSquared() const noexcept;

	/**
	* @brief Computes the Manhattan norm: \f$\sum |x_i|\f$.
	*
	* @return Manhattan norm value as type @p T.
	*/
	[[nodiscard]] T NormL1() const noexcept;

	/**
	* @brief Computes the Chebyshev norm: \f$\max\abs{x_i}\f$.
	*
	* @return Chebyshev norm value as type @p T.
	*/
	[[nodiscard]] T NormLInf() const noexcept;

	/**
	* @brief Normalises a vector into a unit vector in the same direction: \f$\hat{v} = \frac{v}{\|v\|_2}\f$.
	*
	* @return A new normalised vector.
	*
	* @throws std::runtime_error if the vector is a zero vector.
	*/
	[[nodiscard]] Vector Normalised() const;

	/**
	* @brief Computes the three-dimensional cross product (enabled when N == 3).
	*
	* \f$(a \times b) = (a_y b_z - a_z b_y,\; a_z b_x - a_x b_z,\; a_x b_y - a_y b_x)\f$
	*
	* @param rhs Right hand side vector.
	*
	* @return A new vector with the cross product applied.
	*/
	template<std::size_t M = N>
	[[nodiscard]] std::enable_if_t<M == 3, Vector> Cross(const Vector& rhs) const noexcept;

	/**
	* @brief Sum of all elements in vector.
	*
	* @return Sum of all elements as type @p T.
	*/
	[[nodiscard]] T Sum() const noexcept;

	/**
	* @brief Min element in vector.
	*
	* @return Min element as type @p T.
	*/
	[[nodiscard]] T Min() const noexcept;

	/**
	* @brief Max element in vector.
	*
	* @return Max element as type @p T.
	*/
	[[nodiscard]] T Max() const noexcept;

	// ----- String conversion -----

	/**
	* @brief Returns a string representation of the vector.
	*
	* @return String in the format `[x, y, z]`.
	*/
	std::string ToString() const;

	/**
	* @brief Returns a multi-line string representation of the vector as a column.
	*
	* @return Multi-line column string, e.g. for `{1, 2, 3}`:
	* @code
	* [ 1 ]
	* [ 2 ]
	* [ 3 ]
	* @endcode
	*/
	std::string Prettify() const;

	/// @brief Stream insertion operator.
	friend std::ostream& operator<<(std::ostream& os, const Vector& v)
	{
		return os << v.ToString();
	}

private:
    // _simd_align: largest SIMD alignment that fits the object's actual byte size.
    //   N*sizeof(T) >= 32 → 32: AVX2 256-bit aligned loads/stores (e.g. Vec4d = 32 B)
    //   N*sizeof(T) >= 16 → 16: SSE 128-bit aligned loads/stores (e.g. Vec4f = 16 B)
    //   N*sizeof(T) <  16 → alignof(T): natural alignment, no padding waste (e.g. Vec2f = 8 B)
    // Without this, std::array only guarantees alignof(T), which prevents the compiler
    // from emitting aligned vector loads and suppresses auto-vectorisation.
    static constexpr std::size_t _simd_align =
        (N * sizeof(T) >= 32) ? 32u :
        (N * sizeof(T) >= 16) ? 16u :
        alignof(T);

    alignas(_simd_align) std::array<T, N> _data; ///< Fixed size array to store elements.
};

// ----- Aliases -----
// These are commonly used vector types predefined for convenience.
using Vec2f = Vector<float,  2>;
using Vec3f = Vector<float,  3>;
using Vec4f = Vector<float,  4>;
using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

} // namespace lumina::core

#include "../detail/core/vector.tpp"