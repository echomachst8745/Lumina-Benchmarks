namespace lumina::core {

// ----- Constructors -----

template<typename T, std::size_t N>
constexpr Vector<T, N>::Vector() noexcept : _data{} {}

template<typename T, std::size_t N>
constexpr Vector<T, N>::Vector(T fill_value) noexcept { _data.fill(fill_value); }

template<typename T, std::size_t N>
constexpr Vector<T, N>::Vector(std::initializer_list<T> init_list)
{
    if (init_list.size() != N)
    {
        throw std::invalid_argument("Initialiser list size must equal N");
    }

    std::copy(init_list.begin(), init_list.end(), _data.begin());
}

// ----- Element Access -----

template<typename T, std::size_t N>
T& Vector<T, N>::at(size_type i)
{
    if (i >= N)
        throw std::out_of_range("Vector index out of range");
    return _data[i];
}

template<typename T, std::size_t N>
const T& Vector<T, N>::at(size_type i) const
{
    if (i >= N)
        throw std::out_of_range("Vector index out of range");
    return _data[i];
}

template<typename T, std::size_t N>
constexpr T& Vector<T, N>::operator[](size_type i) noexcept
{
    return _data[i];
}

template<typename T, std::size_t N>
constexpr const T& Vector<T, N>::operator[](size_type i) const noexcept
{
    return _data[i];
}

template<typename T, std::size_t N>
constexpr T* Vector<T, N>::data() noexcept
{
    return _data.data();
}

template<typename T, std::size_t N>
constexpr const T* Vector<T, N>::data() const noexcept
{
    return _data.data();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::size_type Vector<T, N>::size() const noexcept
{
    return N;
}

// ----- Iterators -----

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::iterator Vector<T, N>::begin() noexcept
{
    return _data.begin();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::iterator Vector<T, N>::end() noexcept
{
    return _data.end();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::const_iterator Vector<T, N>::begin() const noexcept
{
    return _data.begin();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::const_iterator Vector<T, N>::end() const noexcept
{
    return _data.end();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::const_iterator Vector<T, N>::cbegin() const noexcept
{
    return _data.cbegin();
}

template<typename T, std::size_t N>
constexpr typename Vector<T, N>::const_iterator Vector<T, N>::cend() const noexcept
{
    return _data.cend();
}

// ----- Arithmetic Operations -----

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::Add(const Vector& other) const noexcept
{
    Vector result;
    for (size_type i = 0; i < N; ++i)
        result._data[i] = _data[i] + other._data[i];
    return result;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::operator+(const Vector& other) const noexcept
{
    return Add(other);
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::Subtract(const Vector& rhs) const noexcept
{
    Vector result;
    for (size_type i = 0; i < N; ++i)
        result._data[i] = _data[i] - rhs._data[i];
    return result;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::operator-(const Vector& rhs) const noexcept
{
    return Subtract(rhs);
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::operator-() const noexcept
{
    Vector result;
    for (size_type i = 0; i < N; ++i)
        result._data[i] = -_data[i];
    return result;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::Multiply(T scalar) const noexcept
{
    Vector result;
    for (size_type i = 0; i < N; ++i)
        result._data[i] = _data[i] * scalar;
    return result;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::operator*(T scalar) const noexcept
{
    return Multiply(scalar);
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::Divide(T scalar) const noexcept
{
    // Precompute the reciprocal so the loop uses multiplication instead of division.
    // FP division has ~20-40 cycle latency on modern CPUs; multiplication is ~4-5 cycles.
    // This turns N slow divisions into one division + N fast multiplies.
    const T inv = T{1} / scalar;
    Vector result;
    for (size_type i = 0; i < N; ++i)
        result._data[i] = _data[i] * inv;
    return result;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> Vector<T, N>::operator/(T scalar) const noexcept
{
    return Divide(scalar);
}

template<typename T, std::size_t N>
constexpr Vector<T, N>& Vector<T, N>::operator+=(const Vector& other) noexcept
{
    for (size_type i = 0; i < N; ++i)
        _data[i] += other._data[i];
    return *this;
}

template<typename T, std::size_t N>
constexpr Vector<T, N>& Vector<T, N>::operator-=(const Vector& rhs) noexcept
{
    for (size_type i = 0; i < N; ++i)
        _data[i] -= rhs._data[i];
    return *this;
}

template<typename T, std::size_t N>
constexpr Vector<T, N>& Vector<T, N>::operator*=(T scalar) noexcept
{
    for (size_type i = 0; i < N; ++i)
        _data[i] *= scalar;
    return *this;
}

template<typename T, std::size_t N>
constexpr Vector<T, N>& Vector<T, N>::operator/=(T scalar) noexcept
{
    // Same reciprocal trick as Divide(): one division, N multiplies.
    const T inv = T{1} / scalar;
    for (size_type i = 0; i < N; ++i)
        _data[i] *= inv;
    return *this;
}

// ----- Comparison Operations -----

template<typename T, std::size_t N>
constexpr bool Vector<T, N>::Equals(const Vector& other) const noexcept
{
    return _data == other._data;
}

template<typename T, std::size_t N>
constexpr bool Vector<T, N>::operator==(const Vector& other) const noexcept
{
    return Equals(other);
}

template<typename T, std::size_t N>
constexpr bool Vector<T, N>::operator!=(const Vector& other) const noexcept
{
    return !Equals(other);
}

// ----- Mathematical Operations -----

template<typename T, std::size_t N>
constexpr T Vector<T, N>::Dot(const Vector& rhs) const noexcept
{
    T result{};
    for (size_type i = 0; i < N; ++i)
        result += _data[i] * rhs._data[i];
    return result;
}

template<typename T, std::size_t N>
T Vector<T, N>::NormSquared() const noexcept
{
    return Dot(*this);
}

template<typename T, std::size_t N>
T Vector<T, N>::Norm() const noexcept
{
    return std::sqrt(NormSquared());
}

template<typename T, std::size_t N>
T Vector<T, N>::NormL1() const noexcept
{
    T result{};
    for (size_type i = 0; i < N; ++i)
        result += std::abs(_data[i]);
    return result;
}

template<typename T, std::size_t N>
T Vector<T, N>::NormLInf() const noexcept
{
    T result = std::abs(_data[0]);
    for (size_type i = 1; i < N; ++i)
        result = std::max(result, std::abs(_data[i]));
    return result;
}

template<typename T, std::size_t N>
Vector<T, N> Vector<T, N>::Normalised() const
{
    const T n = Norm();
    if (n == T{})
        throw std::runtime_error("Cannot normalise a zero vector");
    return Divide(n);
}

template<typename T, std::size_t N>
template<std::size_t M>
std::enable_if_t<M == 3, Vector<T, N>> Vector<T, N>::Cross(const Vector& rhs) const noexcept
{
    return Vector{
        _data[1] * rhs._data[2] - _data[2] * rhs._data[1],
        _data[2] * rhs._data[0] - _data[0] * rhs._data[2],
        _data[0] * rhs._data[1] - _data[1] * rhs._data[0]
    };
}

template<typename T, std::size_t N>
T Vector<T, N>::Sum() const noexcept
{
    T result{};
    for (size_type i = 0; i < N; ++i)
        result += _data[i];
    return result;
}

template<typename T, std::size_t N>
T Vector<T, N>::Min() const noexcept
{
    return *std::min_element(_data.begin(), _data.end());
}

template<typename T, std::size_t N>
T Vector<T, N>::Max() const noexcept
{
    return *std::max_element(_data.begin(), _data.end());
}

// ----- String Conversion -----

template<typename T, std::size_t N>
std::string Vector<T, N>::ToString() const
{
    std::ostringstream oss;
    oss << '[';
    for (size_type i = 0; i < N; ++i)
    {
        if (i > 0) oss << ", ";
        oss << _data[i];
    }
    oss << ']';
    return oss.str();
}

template<typename T, std::size_t N>
std::string Vector<T, N>::Prettify() const
{
    int width = 0;
    for (const auto& v : _data)
    {
        std::ostringstream tmp;
        tmp << v;
        width = std::max(width, static_cast<int>(tmp.str().size()));
    }

    std::ostringstream oss;
    for (size_type i = 0; i < N; ++i)
    {
        oss << "[ " << std::setw(width) << _data[i] << " ]";
        if (i + 1 < N) oss << '\n';
    }
    return oss.str();
}

} // namespace lumina::core
