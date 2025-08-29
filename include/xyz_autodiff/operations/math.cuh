#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>

namespace xyz_autodiff {
namespace math {

// =============================================================================
// Exponential functions
// =============================================================================

template <typename T>
__host__ __device__ inline T exp(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return expf(x);
        #else
        return std::exp(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::exp(x);  // CUDA double exp
        #else
        return std::exp(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T log(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return logf(x);
        #else
        return std::log(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::log(x);  // CUDA double log
        #else
        return std::log(x);
        #endif
    }
}

// =============================================================================
// Trigonometric functions
// =============================================================================

template <typename T>
__host__ __device__ inline T sin(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return sinf(x);
        #else
        return std::sin(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::sin(x);
        #else
        return std::sin(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T cos(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return cosf(x);
        #else
        return std::cos(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::cos(x);
        #else
        return std::cos(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T tan(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return tanf(x);
        #else
        return std::tan(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::tan(x);
        #else
        return std::tan(x);
        #endif
    }
}

// =============================================================================
// Hyperbolic functions
// =============================================================================

template <typename T>
__host__ __device__ inline T sinh(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return sinhf(x);
        #else
        return std::sinh(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::sinh(x);
        #else
        return std::sinh(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T cosh(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return coshf(x);
        #else
        return std::cosh(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::cosh(x);
        #else
        return std::cosh(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T tanh(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return tanhf(x);
        #else
        return std::tanh(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::tanh(x);
        #else
        return std::tanh(x);
        #endif
    }
}

// =============================================================================
// Power and root functions
// =============================================================================

template <typename T>
__host__ __device__ inline T sqrt(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return sqrtf(x);
        #else
        return std::sqrt(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::sqrt(x);
        #else
        return std::sqrt(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T pow(const T& base, const T& exponent) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return powf(base, exponent);
        #else
        return std::pow(base, exponent);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return ::pow(base, exponent);
        #else
        return std::pow(base, exponent);
        #endif
    }
}

// =============================================================================
// Activation functions (commonly used in neural networks)
// =============================================================================

template <typename T>
__host__ __device__ inline T sigmoid(const T& x) {
    const T one = T{1};
    return one / (one + exp(-x));
}

template <typename T>
__host__ __device__ inline T relu(const T& x) {
    const T zero = T{0};
    return x > zero ? x : zero;
}

template <typename T>
__host__ __device__ inline T leaky_relu(const T& x, const T& alpha = T{0.01}) {
    const T zero = T{0};
    return x > zero ? x : alpha * x;
}

template <typename T>
__host__ __device__ inline T elu(const T& x, const T& alpha = T{1}) {
    const T zero = T{0};
    const T one = T{1};
    return x >= zero ? x : alpha * (exp(x) - one);
}

template <typename T>
__host__ __device__ inline T softplus(const T& x) {
    return log(T{1} + exp(x));
}

template <typename T>
__host__ __device__ inline T swish(const T& x) {
    return x * sigmoid(x);
}

template <typename T>
__host__ __device__ inline T gelu(const T& x) {
    const T half = T{0.5};
    const T one = T{1};
    const T sqrt_2_pi = T{0.7978845608028654}; // sqrt(2/pi)
    const T coeff = T{0.044715};
    
    return half * x * (one + tanh(sqrt_2_pi * (x + coeff * pow(x, T{3}))));
}

// =============================================================================
// Utility functions
// =============================================================================

template <typename T>
__host__ __device__ inline T abs(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return fabsf(x);
        #else
        return std::abs(x);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return fabs(x);
        #else
        return std::abs(x);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T max(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return fmaxf(a, b);
        #else
        return std::max(a, b);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return fmax(a, b);
        #else
        return std::max(a, b);
        #endif
    }
}

template <typename T>
__host__ __device__ inline T min(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, float>) {
        #ifdef __CUDA_ARCH__
        return fminf(a, b);
        #else
        return std::min(a, b);
        #endif
    } else if constexpr (std::is_same_v<T, double>) {
        #ifdef __CUDA_ARCH__
        return fmin(a, b);
        #else
        return std::min(a, b);
        #endif
    }
}

} // namespace math
} // namespace xyz_autodiff