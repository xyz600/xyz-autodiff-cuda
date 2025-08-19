#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/variable.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

class VariableOperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA device availability
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// Test buffer for Variable + constant operator
template <typename T, std::size_t N>
struct VariableOperatorTestBuffer {
    T input_data[N];
    T input_grad[N];
    T output_data[N];
    T constant_value;
    T output_grad[N];
};

// CUDA kernel to test Variable + constant operator
template <typename T, std::size_t N>
__global__ void test_variable_plus_constant_kernel(VariableOperatorTestBuffer<T, N>* buffer) {
    // Create VariableRef from buffer data and gradients
    xyz_autodiff::VariableRef<T, N> input(buffer->input_data, buffer->input_grad);
    
    // Zero gradient
    input.zero_grad();
    
    // Create operation and forward
    auto result_op = input + buffer->constant_value;
    result_op.forward();
    
    // Copy result to output data array
    for (std::size_t i = 0; i < N; ++i) {
        buffer->output_data[i] = result_op[i];
    }
    
    // Set gradient on output
    for (std::size_t i = 0; i < N; ++i) {
        result_op.add_grad(i, buffer->output_grad[i]);
    }
    
    // Backward pass
    result_op.backward();
}

TEST_F(VariableOperatorTest, VariablePlusConstantFloat) {
    using T = float;
    constexpr std::size_t N = 3;
    
    // Prepare host data
    VariableOperatorTestBuffer<T, N> host_buffer;
    
    // Initialize input values on host
    T input_values[N] = {1.0f, 2.0f, 3.0f};
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_data[i] = input_values[i];
    }
    
    // Set constant value
    host_buffer.constant_value = 2.5f;
    
    // Initialize input gradients to zero
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_grad[i] = 0.0f;
    }
    
    // Set output gradients (all 1.0 for simplicity)
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.output_grad[i] = 1.0f;
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableOperatorTestBuffer<T, N>>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_plus_constant_kernel<T, N><<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyDeviceToHost);
    
    // Verify results on host
    T expected_output[N] = {1.0f + 2.5f, 2.0f + 2.5f, 3.0f + 2.5f};
    T expected_input_grad[N] = {1.0f, 1.0f, 1.0f}; // Since d/dx(x + c) = 1
    
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], expected_output[i], 1e-6f) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.input_grad[i], expected_input_grad[i], 1e-6f)
            << "Backward pass failed at index " << i;
    }
}

TEST_F(VariableOperatorTest, VariablePlusConstantDouble) {
    using T = double;
    constexpr std::size_t N = 4;
    
    // Prepare host data
    VariableOperatorTestBuffer<T, N> host_buffer;
    
    // Initialize input values on host
    T input_values[N] = {1.0, 2.0, 3.0, 4.0};
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_data[i] = input_values[i];
    }
    
    // Set constant value
    host_buffer.constant_value = -1.25;
    
    // Initialize input gradients to zero
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_grad[i] = 0.0;
    }
    
    // Set output gradients (all 1.0 for simplicity)
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.output_grad[i] = 1.0;
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableOperatorTestBuffer<T, N>>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_plus_constant_kernel<T, N><<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyDeviceToHost);
    
    // Verify results on host
    T expected_output[N] = {1.0 + (-1.25), 2.0 + (-1.25), 3.0 + (-1.25), 4.0 + (-1.25)};
    T expected_input_grad[N] = {1.0, 1.0, 1.0, 1.0}; // Since d/dx(x + c) = 1
    
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], expected_output[i], 1e-10) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.input_grad[i], expected_input_grad[i], 1e-10)
            << "Backward pass failed at index " << i;
    }
}

// Test with zero constant
TEST_F(VariableOperatorTest, VariablePlusZeroConstant) {
    using T = float;
    constexpr std::size_t N = 2;
    
    // Prepare host data
    VariableOperatorTestBuffer<T, N> host_buffer;
    
    // Initialize input values on host
    T input_values[N] = {1.0f, 2.0f};
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_data[i] = input_values[i];
    }
    
    // Set constant value to zero
    host_buffer.constant_value = 0.0f;
    
    // Initialize input gradients to zero
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.input_grad[i] = 0.0f;
    }
    
    // Set output gradients (all 1.0 for simplicity)
    for (std::size_t i = 0; i < N; ++i) {
        host_buffer.output_grad[i] = 1.0f;
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableOperatorTestBuffer<T, N>>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_plus_constant_kernel<T, N><<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableOperatorTestBuffer<T, N>), cudaMemcpyDeviceToHost);
    
    // Verify results on host (should be same as input since adding zero)
    T expected_output[N] = {1.0f, 2.0f};
    T expected_input_grad[N] = {1.0f, 1.0f}; // Since d/dx(x + c) = 1
    
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], expected_output[i], 1e-6f) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.input_grad[i], expected_input_grad[i], 1e-6f)
            << "Backward pass failed at index " << i;
    }
}