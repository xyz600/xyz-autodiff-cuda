#pragma once

namespace xyz_autodiff {

template <typename T, int N>
struct ConstArray {
    using value_type = T;

    T data[N];

    __device__ __host__ constexpr ConstArray() = default;
    
    __device__ __host__ constexpr ConstArray(const T (&init)[N]) {
        for (int i = 0; i < N; ++i) {
            data[i] = init[i];
        }
    }

    __device__ __host__ constexpr T &operator[](std::size_t index) noexcept {
        return data[index];
    }

    __device__ __host__ constexpr const T &operator[](std::size_t index) const noexcept {
        return data[index];
    }

    __device__ __host__ constexpr int size() const {
        return N;
    }

    // Compound assignment operators
    __device__ __host__ constexpr ConstArray& operator+=(const ConstArray& other) {
        for (int i = 0; i < N; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    __device__ __host__ constexpr ConstArray& operator-=(const ConstArray& other) {
        for (int i = 0; i < N; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    __device__ __host__ constexpr ConstArray& operator*=(const ConstArray& other) {
        for (int i = 0; i < N; ++i) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    __device__ __host__ constexpr ConstArray& operator/=(const ConstArray& other) {
        for (int i = 0; i < N; ++i) {
            data[i] /= other.data[i];
        }
        return *this;
    }
};

// Binary arithmetic operators for ConstArray
template <typename T, int N>
__device__ __host__ constexpr ConstArray<T, N> operator+(const ConstArray<T, N>& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template <typename T, int N>
__device__ __host__ constexpr ConstArray<T, N> operator-(const ConstArray<T, N>& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

template <typename T, int N>
__device__ __host__ constexpr ConstArray<T, N> operator*(const ConstArray<T, N>& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <typename T, int N>
__device__ __host__ constexpr ConstArray<T, N> operator/(const ConstArray<T, N>& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
};

} // namespace xyz_autodiff