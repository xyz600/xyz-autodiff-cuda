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
