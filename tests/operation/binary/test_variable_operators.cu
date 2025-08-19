#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../../../include/variable.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

class VariableToVariableOperatorTest : public ::testing::Test {
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

// Test buffer for Variable + Variable operator
struct VariableAddTestBuffer {
    float var1_data[3];
    float var1_grad[3];
    float var2_data[3];
    float var2_grad[3];
    float result_data[3];
    float result_grad[3];
    float output_grad[3];
};

// CUDA kernel to test Variable + Variable operator
__global__ void test_variable_add_variable_kernel(VariableAddTestBuffer* buffer) {
    // Create VariableRef from buffer data and gradients
    VariableRef<3, float> var1(buffer->var1_data, buffer->var1_grad);
    VariableRef<3, float> var2(buffer->var2_data, buffer->var2_grad);
    
    // Zero gradients
    var1.zero_grad();
    var2.zero_grad();
    
    // Create operation and forward
    auto result_op = var1 + var2;
    result_op.forward();
    
    // Copy result to output data array
    for (std::size_t i = 0; i < 3; ++i) {
        buffer->result_data[i] = result_op[i];
    }
    
    // Set gradient on output
    for (std::size_t i = 0; i < 3; ++i) {
        result_op.add_grad(i, buffer->output_grad[i]);
    }
    
    // Backward pass
    result_op.backward();
}

TEST_F(VariableToVariableOperatorTest, VariableAddVariableFloat) {
    // Prepare host data
    VariableAddTestBuffer host_buffer;
    
    // Initialize input values on host
    float var1_values[3] = {1.0f, 2.0f, 3.0f};
    float var2_values[3] = {4.0f, 5.0f, 6.0f};
    for (std::size_t i = 0; i < 3; ++i) {
        host_buffer.var1_data[i] = var1_values[i];
        host_buffer.var2_data[i] = var2_values[i];
        host_buffer.var1_grad[i] = 0.0f;
        host_buffer.var2_grad[i] = 0.0f;
        host_buffer.output_grad[i] = 1.0f;  // All gradients set to 1
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableAddTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableAddTestBuffer), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_add_variable_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableAddTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify results on host
    float expected_result[3] = {5.0f, 7.0f, 9.0f}; // 1+4, 2+5, 3+6
    float expected_grad[3] = {1.0f, 1.0f, 1.0f};  // d/dx(x + y) = 1 for both x and y
    
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(host_buffer.result_data[i], expected_result[i], 1e-6f) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var1_grad[i], expected_grad[i], 1e-6f)
            << "Var1 backward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var2_grad[i], expected_grad[i], 1e-6f)
            << "Var2 backward pass failed at index " << i;
    }
}

// Test buffer for Variable - Variable operator
struct VariableSubTestBuffer {
    double var1_data[2];
    double var1_grad[2];
    double var2_data[2];
    double var2_grad[2];
    double result_data[2];
    double result_grad[2];
    double output_grad[2];
};

// CUDA kernel to test Variable - Variable operator
__global__ void test_variable_sub_variable_kernel(VariableSubTestBuffer* buffer) {
    // Create VariableRef from buffer data and gradients
    VariableRef<2, double> var1(buffer->var1_data, buffer->var1_grad);
    VariableRef<2, double> var2(buffer->var2_data, buffer->var2_grad);
    
    // Zero gradients
    var1.zero_grad();
    var2.zero_grad();
    
    // Create operation and forward
    auto result_op = var1 - var2;
    result_op.forward();
    
    // Copy result to output data array
    for (std::size_t i = 0; i < 2; ++i) {
        buffer->result_data[i] = result_op[i];
    }
    
    // Set gradient on output
    for (std::size_t i = 0; i < 2; ++i) {
        result_op.add_grad(i, buffer->output_grad[i]);
    }
    
    // Backward pass
    result_op.backward();
}

TEST_F(VariableToVariableOperatorTest, VariableSubVariableDouble) {
    // Prepare host data
    VariableSubTestBuffer host_buffer;
    
    // Initialize input values on host
    double var1_values[2] = {10.0, 15.0};
    double var2_values[2] = {3.0, 5.0};
    for (std::size_t i = 0; i < 2; ++i) {
        host_buffer.var1_data[i] = var1_values[i];
        host_buffer.var2_data[i] = var2_values[i];
        host_buffer.var1_grad[i] = 0.0;
        host_buffer.var2_grad[i] = 0.0;
        host_buffer.output_grad[i] = 1.0;  // All gradients set to 1
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableSubTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableSubTestBuffer), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_sub_variable_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableSubTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify results on host
    double expected_result[2] = {7.0, 10.0}; // 10-3, 15-5
    double expected_var1_grad[2] = {1.0, 1.0};  // d/dx(x - y) = 1 for x
    double expected_var2_grad[2] = {-1.0, -1.0};  // d/dy(x - y) = -1 for y
    
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(host_buffer.result_data[i], expected_result[i], 1e-10) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var1_grad[i], expected_var1_grad[i], 1e-10)
            << "Var1 backward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var2_grad[i], expected_var2_grad[i], 1e-10)
            << "Var2 backward pass failed at index " << i;
    }
}

// Test with operation chain results (using generic operators)
struct VariableChainTestBuffer {
    float var1_data[2];
    float var1_grad[2];
    float var2_data[2];
    float var2_grad[2];
    float var3_data[2];
    float var3_grad[2];
    float result_data[2];
    float result_grad[2];
    float output_grad[2];
};

// CUDA kernel to test operation chain: (var1 + var2) - var3
__global__ void test_variable_chain_operations_kernel(VariableChainTestBuffer* buffer) {
    // Create VariableRef from buffer data and gradients
    VariableRef<2, float> var1(buffer->var1_data, buffer->var1_grad);
    VariableRef<2, float> var2(buffer->var2_data, buffer->var2_grad);
    VariableRef<2, float> var3(buffer->var3_data, buffer->var3_grad);
    
    // Zero gradients
    var1.zero_grad();
    var2.zero_grad();
    var3.zero_grad();
    
    // Create operation chain: (var1 + var2) - var3
    auto intermediate = var1 + var2;  // Uses Variable + Variable operator
    auto result_op = intermediate - var3;  // Uses generic operator (operation result - Variable)
    
    result_op.forward();
    
    // Copy result to output data array
    for (std::size_t i = 0; i < 2; ++i) {
        buffer->result_data[i] = result_op[i];
    }
    
    // Set gradient on output
    for (std::size_t i = 0; i < 2; ++i) {
        result_op.add_grad(i, buffer->output_grad[i]);
    }
    
    // Backward pass
    result_op.backward();
}

TEST_F(VariableToVariableOperatorTest, VariableOperationChain) {
    // Prepare host data
    VariableChainTestBuffer host_buffer;
    
    // Initialize input values on host
    float var1_values[2] = {2.0f, 3.0f};
    float var2_values[2] = {4.0f, 5.0f};
    float var3_values[2] = {1.0f, 2.0f};
    
    for (std::size_t i = 0; i < 2; ++i) {
        host_buffer.var1_data[i] = var1_values[i];
        host_buffer.var2_data[i] = var2_values[i];
        host_buffer.var3_data[i] = var3_values[i];
        host_buffer.var1_grad[i] = 0.0f;
        host_buffer.var2_grad[i] = 0.0f;
        host_buffer.var3_grad[i] = 0.0f;
        host_buffer.output_grad[i] = 1.0f;
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableChainTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableChainTestBuffer), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_chain_operations_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableChainTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify results: (var1 + var2) - var3 = (2+4, 3+5) - (1, 2) = (6, 8) - (1, 2) = (5, 6)
    float expected_result[2] = {5.0f, 6.0f};
    float expected_grad[2] = {1.0f, 1.0f};  // All variables have gradient 1 for var1,var2 and -1 for var3
    float expected_var3_grad[2] = {-1.0f, -1.0f};  // var3 has negative gradient
    
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(host_buffer.result_data[i], expected_result[i], 1e-6f) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var1_grad[i], expected_grad[i], 1e-6f)
            << "Var1 backward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var2_grad[i], expected_grad[i], 1e-6f)
            << "Var2 backward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var3_grad[i], expected_var3_grad[i], 1e-6f)
            << "Var3 backward pass failed at index " << i;
    }
}

// Test with zero values
TEST_F(VariableToVariableOperatorTest, VariableAddZeroValues) {
    // Prepare host data
    VariableAddTestBuffer host_buffer;
    
    // Initialize input values on host (all zeros)
    for (std::size_t i = 0; i < 3; ++i) {
        host_buffer.var1_data[i] = 0.0f;
        host_buffer.var2_data[i] = 0.0f;
        host_buffer.var1_grad[i] = 0.0f;
        host_buffer.var2_grad[i] = 0.0f;
        host_buffer.output_grad[i] = 1.0f;
    }
    
    // Copy to device
    auto device_buffer = makeCudaUnique<VariableAddTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(VariableAddTestBuffer), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_variable_add_variable_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(VariableAddTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify results (should be all zeros for forward, 1s for gradients)
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(host_buffer.result_data[i], 0.0f, 1e-6f) 
            << "Forward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var1_grad[i], 1.0f, 1e-6f)
            << "Var1 backward pass failed at index " << i;
        EXPECT_NEAR(host_buffer.var2_grad[i], 1.0f, 1e-6f)
            << "Var2 backward pass failed at index " << i;
    }
}