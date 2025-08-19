# Mini Gaussian Splatting with CUDA Automatic Differentiation

This example demonstrates a simplified implementation of Gaussian splatting using the xyz-autodiff CUDA automatic differentiation framework. It showcases various mathematical operations needed for 2D Gaussian splatting with full gradient computation.

## Overview

Gaussian splatting is a technique for rendering 3D scenes using 2D Gaussian distributions. This mini implementation focuses on the core mathematical operations required for evaluating a 2D Gaussian at a query point and computing gradients with respect to all Gaussian parameters.

## Implemented Operations

### Core Mathematical Operations

1. **Matrix Multiplication** (`matrix_multiplication.cuh`)
   - 3x3 matrix multiplication with gradient support
   - Used for transforming coordinate systems

2. **Element-wise Exponential** (`element_wise_exp.cuh`)
   - Standard element-wise exp(x) operation
   - Specialized exp(-x) operation for Gaussian evaluation
   - Both with analytical gradients

3. **Norm Operations** (`norm_operations.cuh`)
   - L1 norm: ||x||₁ = Σ|xᵢ|
   - L2 norm: ||x||₂ = √(Σxᵢ²)
   - L2 squared norm: ||x||₂² = Σxᵢ²
   - All with proper gradient handling

4. **Norm Combination** (`norm_addition.cuh`)
   - L1 + L2 norm regularization
   - Weighted L1 + L2 combinations
   - Scalar addition for combining results

### Gaussian Splatting Specific Operations

5. **Covariance Matrix Generation** (`covariance_generation.cuh`)
   - Generate 2x2 covariance matrix from scale and rotation parameters
   - M = R × S where R is rotation matrix, S is diagonal scale matrix
   - Σ = M × Mᵀ covariance computation
   - Direct scale/rotation to 3-parameter symmetric matrix conversion

6. **Symmetric Matrix Operations** (`symmetric_matrix.cuh`)
   - 3-parameter representation of 2x2 symmetric matrices: [a,b,c] → [[a,b],[b,c]]
   - Matrix inverse computation for 2x2 symmetric matrices
   - Conversion between full matrix and compact representations

7. **Mahalanobis Distance** (`mahalanobis_distance.cuh`)
   - Distance computation: (x-μ)ᵀ Σ⁻¹ (x-μ)
   - Version with explicit difference vector
   - Version with separate query point and center
   - Essential for Gaussian evaluation

8. **Element-wise Multiplication** (`element_wise_multiply.cuh`)
   - 2-input element-wise multiplication
   - 3-input element-wise multiplication (c × d × o operation)
   - Scalar multiplication
   - Used for combining colors, opacity, and Gaussian values

## Mathematical Pipeline

The complete Gaussian splatting evaluation follows this pipeline:

1. **Covariance Generation**: Convert scale [sx, sy] and rotation θ to covariance matrix Σ
2. **Matrix Inversion**: Compute Σ⁻¹ for Mahalanobis distance
3. **Distance Computation**: Calculate (x-μ)ᵀ Σ⁻¹ (x-μ)
4. **Gaussian Evaluation**: Compute exp(-0.5 × distance²)
5. **Color Blending**: Apply opacity and color: color × opacity × gaussian_value
6. **Final Result**: Combine with norm operations for regularization

## Files Structure

```
examples/mini-gaussian-splatting/
├── CMakeLists.txt                     # Build configuration
├── README.md                          # This documentation
├── mini_gaussian_splatting_example.cu # Complete working example
├── matrix_multiplication.cuh          # 3x3 matrix operations
├── element_wise_exp.cuh              # Exponential operations
├── norm_operations.cuh               # L1/L2 norm computations
├── norm_addition.cuh                 # Norm combinations
├── symmetric_matrix.cuh              # Symmetric matrix operations
├── covariance_generation.cuh         # Covariance matrix generation
├── mahalanobis_distance.cuh          # Distance computations
├── element_wise_multiply.cuh         # Multiplication operations
└── tests/                            # Comprehensive test suite
    ├── test_matrix_operations.cu     # Matrix operation tests
    ├── test_norm_operations.cu       # Norm operation tests
    └── test_gaussian_splatting.cu    # Core Gaussian splatting tests
```

## Usage

### Building

From the project root directory:

```bash
task build:debug
cd build/debug
```

### Running the Example

```bash
./examples/mini-gaussian-splatting/mini_gaussian_splatting_example
```

### Running Tests

```bash
# Run all mini Gaussian splatting tests
./test_mini_gaussian_matrix_operations
./test_mini_gaussian_norm_operations  
./test_mini_gaussian_splatting

# Or run through CTest
ctest -R MiniGaussian
```

## Example Output

The example evaluates a 2D Gaussian with:
- Center: (0, 0)
- Scale: (1.0, 0.5) 
- Rotation: 0.1 radians
- Color: (1.0, 0.5, 0.2)
- Opacity: 0.8
- Query point: (0.5, 0.3)

It outputs all intermediate results and gradients with respect to all parameters.

## Key Features

1. **Full Gradient Support**: All operations compute analytical gradients
2. **Numerical Verification**: Tests verify gradients using numerical differentiation
3. **DAG Computation**: Complex operation chaining with proper gradient flow
4. **Memory Efficient**: Uses compact 3-parameter representation for symmetric matrices
5. **CUDA Optimized**: All operations run on GPU with proper device/host annotations

## Mathematical Background

### 2D Gaussian Function

The 2D Gaussian is defined as:

```
G(x) = exp(-0.5 × (x-μ)ᵀ Σ⁻¹ (x-μ))
```

Where:
- x is the query point
- μ is the center
- Σ is the 2x2 covariance matrix

### Covariance Matrix Parameterization

Instead of directly optimizing the 2x2 covariance matrix Σ, we use:
- Scale parameters: [sx, sy]
- Rotation angle: θ

The covariance is constructed as: Σ = R(θ) × S × R(θ)ᵀ

Where:
- R(θ) is the 2D rotation matrix
- S = diag(sx², sy²) is the diagonal scale matrix

This ensures the covariance matrix remains positive definite during optimization.

## Integration with xyz-autodiff

This example demonstrates advanced usage of the xyz-autodiff framework:

- Custom operation definitions using the Logic/Operation pattern
- Complex operation chaining with automatic gradient propagation
- Memory management with cuda_unique_ptr
- Concept-based template programming for type safety
- Integration with existing operations (quaternion to rotation matrix)

The implementation serves as both a practical example and a test case for the framework's capabilities in handling complex mathematical computations with automatic differentiation.