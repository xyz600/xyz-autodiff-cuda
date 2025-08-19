# Claude Development Notes

## Build Commands

### Clean Build
```bash
task clean
```

### Debug Build
```bash
task build:debug
```

### Release Build
```bash
task build:release
```

## Testing

### Run All Tests
```bash
task test
```

### Run Specific Tests
```bash
# Variable tests
task build:debug && cd build/debug && ./tests/test_variable

# DenseMatrix tests  
task build:debug && cd build/debug && ./tests/test_dense_matrix

# Parallel gradient accumulation tests (atomicAdd)
task build:debug && cd build/debug && ./tests/test_parallel_gradient_accumulation
```

## Key Features

### Thread-Safe Gradient Accumulation
The system supports concurrent gradient accumulation using `add_grad()`:

- **VariableRef**: Uses atomicAdd for shared/global memory, regular addition for local/register memory
- **Variable**: Uses regular addition (single-threaded access assumed)
- **Automatic detection**: `__isShared()` and `__isGlobal()` determine memory space and choose appropriate method

### Memory Management for Tests
Tests should use `cuda_unique_ptr` instead of direct `cudaMalloc`/`cudaFree`:

```cpp
#include "../include/util/cuda_unique_ptr.cuh"

// Instead of: cudaMalloc(&ptr, sizeof(T))
auto device_ptr = makeCudaUnique<T>();

// Use with: device_ptr.get()
// Automatic cleanup when scope ends
```

#### CRITICAL CONSTRAINT: Single GPU Memory Allocation Per Test
**Each TEST_F function MUST allocate GPU memory only ONCE per test method.**

- Only one call to `makeCudaUnique*()`, `makeCudaUniqueArray()`, or indirect `cudaMalloc` functions per TEST_F
- Use unified buffer structures to manage all test data in a single allocation
- Create test-specific buffer structures when multiple parameters are needed:

#### ADDITIONAL CONSTRAINT: Direct cudaMalloc/cudaFree Prohibition
**Direct calls to `cudaMalloc` and `cudaFree` are PROHIBITED in all test files within `tests/` directory.**

- Use `makeCudaUnique*()` functions for automatic memory management
- This ensures consistent memory management patterns and prevents memory leaks

```cpp
// GOOD: Single allocation with unified structure
template <typename T>
struct TestBuffers {
    T input_data[N];
    T input_grad[N];
    T output_data[M];
    T output_grad[M];
    T expected_results[M];
};
auto device_buffers = makeCudaUnique<TestBuffers<T>>();

// BAD: Multiple allocations
auto device_input = makeCudaUniqueArray<T>(N);   // ❌ 
auto device_output = makeCudaUniqueArray<T>(M);  // ❌ Second allocation

// PROHIBITED: Direct cudaMalloc/cudaFree calls
float* device_data;
cudaMalloc(&device_data, sizeof(float));         // ❌ PROHIBITED
// ... usage ...
cudaFree(device_data);                          // ❌ PROHIBITED
```

This constraint ensures:
- Efficient memory usage
- Predictable memory management 
- Simplified debugging and profiling
- Consistent test architecture

### Gradient Clearing Best Practices  
Use minimal `zero_grad()` calls in computation graphs:

```cpp
// GOOD: Single zero_grad() call from output propagates through graph
final_result.zero_grad();
final_result.add_grad(0, 1.0);
final_result.backward();  // Automatically propagates to inputs

// AVOID: Multiple zero_grad() calls
// final_result.zero_grad();
// input1.zero_grad();  // Redundant
// input2.zero_grad();  // Redundant  
```

### Testing
- **test_parallel_gradient_accumulation**: Tests 10,000 concurrent threads accumulating gradients
- **test_shared_memory_atomic**: Tests shared memory atomicAdd functionality
- **All tests pass**: 100% success rate with 13 test suites

## Examples

### Linear Regression SGD Optimization
A comprehensive stochastic gradient descent example for parameter estimation:

```bash
# Build and run linear regression example
task build:debug && cd build/debug && ./examples/linear_regression_sgd
```

**Features:**
- Fits parameters (a,b,c,d) for function: `y = (x1-a)² + b(x2-c)² + d`
- 100,000 noisy training samples
- 1,000 epochs with 10,000 random samples per batch
- Exponential learning rate decay (0.01 → 0.0001)
- Real-time parameter error tracking

**Current Status:**
- Successfully demonstrates automatic differentiation in CUDA
- Shows gradient accumulation across batches
- Partial convergence achieved (needs optimization tuning for full convergence)

## Operation実行の指針
- **Operationチェーンの構築**: 各operationのファクトリ関数（op::add, op::mul, l1_norm, l2_norm等）で操作グラフを構築する時は、`.forward()`を呼ばない
- **最終実行**: 最後の出力operationで一度だけ`.run()`を呼ぶ
- **run()の動作**: `.run()`は内部で`.forward()`（前向き計算）と`.backward()`（勾配計算）の両方を実行する
- **例**:
  ```cpp
  // ❌ 各operation毎にforward()を呼ぶのは避ける
  auto a = op::add(x, y); a.forward();
  auto b = op::mul(a, z); b.forward();
  b.run();
  
  // ✅ 正しい: operationチェーンを作成してから最後にrun()
  auto a = op::add(x, y);
  auto b = op::mul(a, z);
  b.run();  // forward + backward を実行
  ```

## 修正済みの問題
- zero_grad()が再帰的に入力をゼロクリアする問題を修正
- 現在は各Operationの出力のみをゼロクリアし、入力は保持

## 残存する問題
- 多くのgradient verification testsが失敗（解析的勾配が数値的勾配の約半分）
- DAGテストのMultiplePathsToSameNodeが失敗（勾配が1.0、期待値4.0）
- 参照カウントメカニズムの動作に課題

## Testing Guidelines for Mini Gaussian Splatting Operations

When creating tests for operations in `examples/mini-gaussian-splatting/`, follow these patterns:

### 1. File Structure
- Create one test file per operation: `test_<operation_name>.cu`
- Place tests in `examples/mini-gaussian-splatting/tests/`

### 2. Test Categories (Required for each operation)

#### A. Static Assert Tests for Concept Compliance
```cpp
// Test that operations satisfy Variable and OperationNode concepts
static_assert(VariableConcept<OperationType>, "...");
static_assert(DifferentiableVariableConcept<OperationType>, "...");
static_assert(OperationNode<OperationType>, "...");
static_assert(!OperationNode<Variable<T, N>>, "Variable should NOT be OperationNode");
```

#### B. Forward Pass Tests
```cpp
// Test correctness of forward computation with known inputs/outputs
__global__ void test_operation_forward_kernel(float* result) { /* ... */ }
TEST_F(OperationTest, ForwardPass) { /* ... */ }
```

#### C. Gradient Verification Tests (Double Precision)
```cpp
// Use utility classes from tests/utility/
TEST_F(OperationTest, GradientVerification) {
    using Logic = OperationLogic<VariableRef<double, N>>;
    test::UnaryGradientTester<Logic, InputDim, OutputDim>::test_custom(
        "OperationName", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}
```

#### D. Specific Gradient Tests
```cpp
// Test specific mathematical properties of gradients
__global__ void test_operation_gradient_kernel(double* result) {
    // Test analytical vs numerical gradients
    // Compare with known mathematical derivatives
}
```

#### E. Interface Compliance Tests
```cpp
// Test that all Variable and OperationNode methods work correctly
__global__ void test_operation_interface_kernel(float* result) {
    // Test: forward(), backward(), zero_grad(), data(), grad(), etc.
}
```

### 3. Utility Usage
- **Unary operations**: Use `test::UnaryGradientTester` from `tests/utility/unary_gradient_tester.cuh`
- **Binary operations**: Use `test::BinaryGradientTester` from `tests/utility/binary_gradient_tester.cuh`  
- **Ternary operations**: Create operations and test manually or extend utilities

### 4. Precision Guidelines
- Use **double precision** for gradient verification tests
- Use appropriate tolerance values:
  - Smooth operations: `1e-5` tolerance, `1e-7` delta
  - Non-smooth operations (L1 norm): `1e-4` tolerance, `1e-6` delta
  - Avoid problematic inputs (e.g., zero for L2 norm)

### 5. CRITICAL TESTING CONSTRAINTS
**TEST SKIPPING IS COMPLETELY FORBIDDEN**
- Never use `GTEST_SKIP()` for any test except CUDA device availability checks
- All gradient verification tests must pass without skipping
- If a test has numerical precision issues, fix the tolerance values or implementation instead of skipping
- If a test has mathematical errors, fix the mathematical logic instead of skipping
- Every operation must have complete test coverage including gradient verification
- Problematic tests must be fixed, not skipped

**TOLERANCE CONSTRAINTS FOR DOUBLE PRECISION TESTS**
- Minimum tolerance for double precision gradient verification tests: `1e-5`
- Setting tolerance values below `1e-5` is COMPLETELY FORBIDDEN for double precision tests
- This constraint prevents numerical precision issues with CUDA double precision operations
- Test utilities automatically FAIL when tolerance values below `1e-5` are used
- Test utilities output maximum error and recommend appropriate tolerance values
- Example acceptable tolerance ranges:
  - Simple operations: `1e-5`
