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

## Forward実行の変更 
- operationのファクトリ関数（op::add, op::mul等）で.forward()を明示的に呼ぶ
- Operation::forward()は input.forward() + 参照カウント → logic.forward() の順で実行
- Operation構築時には自動実行しない（明示的な.forward()呼び出しが必要）

## 修正済みの問題
- zero_grad()が再帰的に入力をゼロクリアする問題を修正
- 現在は各Operationの出力のみをゼロクリアし、入力は保持

## 残存する問題
- 多くのgradient verification testsが失敗（解析的勾配が数値的勾配の約半分）
- DAGテストのMultiplePathsToSameNodeが失敗（勾配が1.0、期待値4.0）
- 参照カウントメカニズムの動作に課題
