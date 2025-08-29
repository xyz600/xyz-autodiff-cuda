#pragma once

#include <xyz_autodiff/concept/const_array.cuh>

namespace xyz_autodiff {

template <typename T, int N>
struct ConstArray {
    using value_type = T;
    static constexpr std::size_t size = N;

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

    // Assignment operator
    __device__ __host__ constexpr ConstArray& operator=(const ConstArray& other) {
        if (this != &other) {
            for (int i = 0; i < N; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Compound assignment operators for same ConstArray type
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

    // Compound assignment operators for ConstArrayLike types
    template<ConstArrayLike Other>
    requires ConstArrayCompatible<ConstArray, Other> && (Other::size == N)
    __device__ constexpr ConstArray& operator+=(const Other& other) {
        for (int i = 0; i < N; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    template<ConstArrayLike Other>
    requires ConstArrayCompatible<ConstArray, Other> && (Other::size == N)
    __device__ constexpr ConstArray& operator-=(const Other& other) {
        for (int i = 0; i < N; ++i) {
            data[i] -= other[i];
        }
        return *this;
    }

    template<ConstArrayLike Other>
    requires ConstArrayCompatible<ConstArray, Other> && (Other::size == N)
    __device__ constexpr ConstArray& operator*=(const Other& other) {
        for (int i = 0; i < N; ++i) {
            data[i] *= other[i];
        }
        return *this;
    }

    template<ConstArrayLike Other>
    requires ConstArrayCompatible<ConstArray, Other> && (Other::size == N)
    __device__ constexpr ConstArray& operator/=(const Other& other) {
        for (int i = 0; i < N; ++i) {
            data[i] /= other[i];
        }
        return *this;
    }
};

// Binary arithmetic operators for ConstArray (same type)
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
}

// Binary arithmetic operators for ConstArray with ConstArrayLike types
template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator+(const ConstArray<T, N>& lhs, const Other& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator+(const Other& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator-(const ConstArray<T, N>& lhs, const Other& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator-(const Other& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator*(const ConstArray<T, N>& lhs, const Other& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator*(const Other& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator/(const ConstArray<T, N>& lhs, const Other& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

template <typename T, int N, ConstArrayLike Other>
requires ConstArrayCompatible<ConstArray<T, N>, Other> && (Other::size == N)
__device__ constexpr ConstArray<T, N> operator/(const Other& lhs, const ConstArray<T, N>& rhs) {
    ConstArray<T, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
};

} // namespace xyz_autodiff