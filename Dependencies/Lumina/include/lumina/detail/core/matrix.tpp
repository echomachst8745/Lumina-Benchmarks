namespace lumina::core {

// ----- Constructors -----

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>::Matrix() noexcept : _data{} {}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>::Matrix(T fill_value) noexcept { _data.fill(fill_value); }

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>::Matrix(std::initializer_list<T> init_list)
{
    if (init_list.size() != N * M)
        throw std::invalid_argument("Initialiser list size must equal N * M");
    std::copy(init_list.begin(), init_list.end(), _data.begin());
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>::Matrix(std::initializer_list<std::initializer_list<T>> init_list_rows)
{
    if (init_list_rows.size() != N)
        throw std::invalid_argument("Row count must equal N");
    size_type row = 0;
    for (const auto& row_list : init_list_rows)
    {
        if (row_list.size() != M)
            throw std::invalid_argument("Column count must equal M");
        std::copy(row_list.begin(), row_list.end(), _data.begin() + row * M);
        ++row;
    }
}

// ----- Common Matrices -----

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Zeros() noexcept
{
    return Matrix{};
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Ones() noexcept
{
    return Matrix(T{1});
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Identity() noexcept
{
    static_assert(N == M, "Identity matrix requires a square matrix");
    Matrix result{};
    for (size_type i = 0; i < N; ++i)
        result._data[i * M + i] = T{1};
    return result;
}

// ----- Dimension getters -----

template<typename T, std::size_t N, std::size_t M>
constexpr typename Matrix<T, N, M>::size_type Matrix<T, N, M>::Rows() noexcept { return N; }

template<typename T, std::size_t N, std::size_t M>
constexpr typename Matrix<T, N, M>::size_type Matrix<T, N, M>::Cols() noexcept { return M; }

template<typename T, std::size_t N, std::size_t M>
constexpr typename Matrix<T, N, M>::size_type Matrix<T, N, M>::size() noexcept { return N * M; }

template<typename T, std::size_t N, std::size_t M>
constexpr bool Matrix<T, N, M>::IsSquare() noexcept { return N == M; }

// ----- Element access -----

template<typename T, std::size_t N, std::size_t M>
T& Matrix<T, N, M>::at(size_type i, size_type j)
{
    if (i >= N || j >= M)
        throw std::out_of_range("Matrix index out of range");
    return _data[i * M + j];
}

template<typename T, std::size_t N, std::size_t M>
const T& Matrix<T, N, M>::at(size_type i, size_type j) const
{
    if (i >= N || j >= M)
        throw std::out_of_range("Matrix index out of range");
    return _data[i * M + j];
}

template<typename T, std::size_t N, std::size_t M>
constexpr T& Matrix<T, N, M>::operator()(size_type i, size_type j) noexcept
{
    return _data[i * M + j];
}

template<typename T, std::size_t N, std::size_t M>
constexpr const T& Matrix<T, N, M>::operator()(size_type i, size_type j) const noexcept
{
    return _data[i * M + j];
}

template<typename T, std::size_t N, std::size_t M>
constexpr T* Matrix<T, N, M>::data() noexcept { return _data.data(); }

template<typename T, std::size_t N, std::size_t M>
constexpr const T* Matrix<T, N, M>::data() const noexcept { return _data.data(); }

// ----- Row / column access -----

template<typename T, std::size_t N, std::size_t M>
std::array<T, M> Matrix<T, N, M>::Row(size_type i) const
{
    std::array<T, M> result;
    std::copy(_data.begin() + i * M, _data.begin() + i * M + M, result.begin());
    return result;
}

template<typename T, std::size_t N, std::size_t M>
std::array<T, N> Matrix<T, N, M>::Col(size_type j) const
{
    std::array<T, N> result;
    for (size_type r = 0; r < N; ++r)
        result[r] = _data[r * M + j];
    return result;
}

template<typename T, std::size_t N, std::size_t M>
void Matrix<T, N, M>::SetRow(size_type i, const std::array<T, M>& row_values)
{
    std::copy(row_values.begin(), row_values.end(), _data.begin() + i * M);
}

template<typename T, std::size_t N, std::size_t M>
void Matrix<T, N, M>::SetCol(size_type j, const std::array<T, N>& col_values)
{
    for (size_type r = 0; r < N; ++r)
        _data[r * M + j] = col_values[r];
}

template<typename T, std::size_t N, std::size_t M>
void Matrix<T, N, M>::SwapRows(size_type i, size_type j) noexcept
{
    if (i == j) return;
    for (size_type k = 0; k < M; ++k)
        std::swap(_data[i * M + k], _data[j * M + k]);
}

template<typename T, std::size_t N, std::size_t M>
void Matrix<T, N, M>::SwapCols(size_type i, size_type j) noexcept
{
    if (i == j) return;
    for (size_type r = 0; r < N; ++r)
        std::swap(_data[r * M + i], _data[r * M + j]);
}

// ----- Arithmetic operations -----

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Add(const Matrix& other) const noexcept
{
    Matrix result;
    for (size_type k = 0; k < N * M; ++k)
        result._data[k] = _data[k] + other._data[k];
    return result;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::operator+(const Matrix& other) const noexcept
{
    return Add(other);
    
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Subtract(const Matrix& rhs) const noexcept
{
    Matrix result;
    for (size_type k = 0; k < N * M; ++k)
        result._data[k] = _data[k] - rhs._data[k];
    return result;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::operator-(const Matrix& rhs) const noexcept
{
    return Subtract(rhs);
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Multiply(T scalar) const noexcept
{
    Matrix result;
    for (size_type k = 0; k < N * M; ++k)
        result._data[k] = _data[k] * scalar;
    return result;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::operator*(T scalar) const noexcept
{
    return Multiply(scalar);
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::Divide(T scalar) const noexcept
{
    // Precompute the reciprocal so the loop uses multiplication instead of division.
    // FP division has ~20-40 cycle latency on modern CPUs; multiplication is ~4-5 cycles.
    // This turns N*M slow divisions into one division + N*M fast multiplies.
    const T inv = T{1} / scalar;
    Matrix result;
    for (size_type k = 0; k < N * M; ++k)
        result._data[k] = _data[k] * inv;
    return result;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M> Matrix<T, N, M>::operator/(T scalar) const noexcept
{
    return Divide(scalar);
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>& Matrix<T, N, M>::operator+=(const Matrix& other) noexcept
{
    for (size_type k = 0; k < N * M; ++k)
        _data[k] += other._data[k];
    return *this;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>& Matrix<T, N, M>::operator-=(const Matrix& rhs) noexcept
{
    for (size_type k = 0; k < N * M; ++k)
        _data[k] -= rhs._data[k];
    return *this;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>& Matrix<T, N, M>::operator*=(T scalar) noexcept
{
    for (size_type k = 0; k < N * M; ++k)
        _data[k] *= scalar;
    return *this;
}

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, N, M>& Matrix<T, N, M>::operator/=(T scalar) noexcept
{
    // Same reciprocal trick as Divide(): one division, N*M multiplies.
    const T inv = T{1} / scalar;
    for (size_type k = 0; k < N * M; ++k)
        _data[k] *= inv;
    return *this;
}

template<typename T, std::size_t N, std::size_t M>
template<std::size_t OtherCols>
constexpr Matrix<T, N, OtherCols> Matrix<T, N, M>::Multiply(const Matrix<T, M, OtherCols>& rhs) const noexcept
{
    Matrix<T, N, OtherCols> result;
    // i-k-j loop order: the inner j-loop strides over a contiguous row of rhs in row-major
    // storage, giving sequential memory access on both operands and letting the compiler
    // auto-vectorise the inner loop.  The naive i-j-k order accesses rhs(k,j) with a
    // stride of OtherCols in the k-loop, thrashing the cache for any non-tiny matrix.
    for (size_type i = 0; i < N; ++i)
        for (size_type k = 0; k < M; ++k)
            for (size_type j = 0; j < OtherCols; ++j)
                result(i, j) += _data[i * M + k] * rhs(k, j);
    return result;
}

template<typename T, std::size_t N, std::size_t M>
template<std::size_t OtherCols>
constexpr Matrix<T, N, OtherCols> Matrix<T, N, M>::operator*(const Matrix<T, M, OtherCols>& rhs) const noexcept
{
    return Multiply(rhs);
}

template<typename T, std::size_t N, std::size_t M>
template<std::size_t VectorSize>
constexpr Vector<T, N> Matrix<T, N, M>::Multiply(const Vector<T, VectorSize>& vec) const
{
    static_assert(VectorSize == M, "Vector size must equal number of matrix columns");
    Vector<T, N> result;
    for (size_type i = 0; i < N; ++i)
        for (size_type j = 0; j < M; ++j)
            result[i] += _data[i * M + j] * vec[j];
    return result;
}

template<typename T, std::size_t N, std::size_t M>
template<std::size_t VectorSize>
constexpr Vector<T, N> Matrix<T, N, M>::operator*(const Vector<T, VectorSize>& vec) const
{
    return Multiply(vec);
}

// ----- Matrix operations -----

template<typename T, std::size_t N, std::size_t M>
constexpr Matrix<T, M, N> Matrix<T, N, M>::Transpose() const noexcept
{
    Matrix<T, M, N> result;
    for (size_type i = 0; i < N; ++i)
        for (size_type j = 0; j < M; ++j)
            result(j, i) = _data[i * M + j];
    return result;
}

template<typename T, std::size_t N, std::size_t M>
T Matrix<T, N, M>::FrobeniusNorm() const noexcept
{
    T sum{};
    for (size_type k = 0; k < N * M; ++k)
        sum += _data[k] * _data[k];
    return std::sqrt(sum);
}

template<typename T, std::size_t N, std::size_t M>
T Matrix<T, N, M>::Trace() const
{
    static_assert(N == M, "Trace requires a square matrix");
    T sum{};
    for (size_type i = 0; i < N; ++i)
        sum += _data[i * M + i];
    return sum;
}

// ----- Matrix properties -----

template<typename T, std::size_t N, std::size_t M>
bool Matrix<T, N, M>::IsSymmetric(T delta) const
{
    if constexpr (N != M) return false;
    for (size_type i = 0; i < N; ++i)
        for (size_type j = i + 1; j < M; ++j)
            if (std::abs(_data[i * M + j] - _data[j * M + i]) > delta)
                return false;
    return true;
}

// ----- Comparison -----

template<typename T, std::size_t N, std::size_t M>
constexpr bool Matrix<T, N, M>::Equals(const Matrix& other) const noexcept
{
    return _data == other._data;
}

template<typename T, std::size_t N, std::size_t M>
constexpr bool Matrix<T, N, M>::operator==(const Matrix& other) const noexcept
{
    return Equals(other);
}

template<typename T, std::size_t N, std::size_t M>
constexpr bool Matrix<T, N, M>::operator!=(const Matrix& other) const noexcept
{
    return !Equals(other);
}

// ----- String Conversion -----

template<typename T, std::size_t N, std::size_t M>
std::string Matrix<T, N, M>::ToString() const
{
    std::ostringstream oss;
    oss << '[';
    for (size_type i = 0; i < N; ++i)
    {
        if (i > 0) oss << ", ";
        oss << '[';
        for (size_type j = 0; j < M; ++j)
        {
            if (j > 0) oss << ", ";
            oss << _data[i * M + j];
        }
        oss << ']';
    }
    oss << ']';
    return oss.str();
}

template<typename T, std::size_t N, std::size_t M>
std::string Matrix<T, N, M>::Prettify() const
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
        oss << "[ ";
        for (size_type j = 0; j < M; ++j)
        {
            if (j > 0) oss << ", ";
            oss << std::setw(width) << _data[i * M + j];
        }
        oss << " ]";
        if (i + 1 < N) oss << '\n';
    }
    return oss.str();
}

} // namespace lumina::core
