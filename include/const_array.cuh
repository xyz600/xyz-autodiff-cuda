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
};

} // namespace xyz_autodiff