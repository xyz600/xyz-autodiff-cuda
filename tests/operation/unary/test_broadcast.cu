#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/unary/broadcast.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

using namespace xyz_autodiff;

class BroadcastOperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// Test buffer for broadcast operator
struct BroadcastTestBuffer {
    float input_data[1];      // Single input value
    float input_grad[1];      // Input gradient
    float output_data[4];     // Broadcasted output (1 -> 4)
    float output_grad[4];     // Output gradients
    float final_grad[4];      // Final output gradients for backward pass
};

// CUDA kernel to test broadcast operation: 1 -> 4
__global__ void test_broadcast_1_to_4_kernel(BroadcastTestBuffer* buffer) {
    VariableRef<1, float> input(buffer->input_data, buffer->input_grad);
    
    input.zero_grad();
    
    // Broadcast from length 1 to length 4
    auto broadcasted = op::broadcast<4>(input);
    broadcasted.forward();
    
    // Copy result to output
    for (std::size_t i = 0; i < 4; ++i) {
        buffer->output_data[i] = broadcasted[i];
    }
    
    // Set gradients on output
    for (std::size_t i = 0; i < 4; ++i) {
        broadcasted.add_grad(i, buffer->final_grad[i]);
    }
    
    // Backward pass
    broadcasted.backward();
}

// Test buffer for larger broadcast: 1 -> 8
struct LargeBroadcastTestBuffer {
    double input_data[1];
    double input_grad[1];
    double output_data[8];
    double output_grad[8];
    double final_grad[8];
};

// CUDA kernel to test larger broadcast operation: 1 -> 8
__global__ void test_broadcast_1_to_8_kernel(LargeBroadcastTestBuffer* buffer) {
    VariableRef<1, double> input(buffer->input_data, buffer->input_grad);
    
    input.zero_grad();
    
    // Broadcast from length 1 to length 8
    auto broadcasted = op::broadcast<8>(input);
    broadcasted.forward();
    
    // Copy result to output
    for (std::size_t i = 0; i < 8; ++i) {
        buffer->output_data[i] = broadcasted[i];
    }
    
    // Set gradients on output
    for (std::size_t i = 0; i < 8; ++i) {
        broadcasted.add_grad(i, buffer->final_grad[i]);
    }
    
    // Backward pass
    broadcasted.backward();
}

TEST_F(BroadcastOperatorTest, Broadcast1To4Float) {
    BroadcastTestBuffer host_buffer;
    
    // Initialize input: single value = 3.5
    host_buffer.input_data[0] = 3.5f;
    host_buffer.input_grad[0] = 0.0f;
    
    // Set output gradients: [1, 2, 3, 4]
    host_buffer.final_grad[0] = 1.0f;
    host_buffer.final_grad[1] = 2.0f;
    host_buffer.final_grad[2] = 3.0f;
    host_buffer.final_grad[3] = 4.0f;
    
    // Copy to device and run
    auto device_buffer = makeCudaUnique<BroadcastTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(BroadcastTestBuffer), cudaMemcpyHostToDevice);
    
    test_broadcast_1_to_4_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(BroadcastTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify forward pass: all output elements should equal input value
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], 3.5f, 1e-6f) 
            << "Forward pass failed at index " << i;
    }
    
    // Verify backward pass: input gradient should be sum of all output gradients
    float expected_input_grad = 1.0f + 2.0f + 3.0f + 4.0f; // = 10.0f
    EXPECT_NEAR(host_buffer.input_grad[0], expected_input_grad, 1e-6f)
        << "Backward pass failed: expected " << expected_input_grad 
        << " but got " << host_buffer.input_grad[0];
}

TEST_F(BroadcastOperatorTest, Broadcast1To8Double) {
    LargeBroadcastTestBuffer host_buffer;
    
    // Initialize input: single value = -2.25
    host_buffer.input_data[0] = -2.25;
    host_buffer.input_grad[0] = 0.0;
    
    // Set output gradients: all equal to 0.5
    for (std::size_t i = 0; i < 8; ++i) {
        host_buffer.final_grad[i] = 0.5;
    }
    
    // Copy to device and run
    auto device_buffer = makeCudaUnique<LargeBroadcastTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(LargeBroadcastTestBuffer), cudaMemcpyHostToDevice);
    
    test_broadcast_1_to_8_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(LargeBroadcastTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify forward pass: all output elements should equal input value
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], -2.25, 1e-10) 
            << "Forward pass failed at index " << i;
    }
    
    // Verify backward pass: input gradient should be sum of all output gradients
    double expected_input_grad = 8 * 0.5; // = 4.0
    EXPECT_NEAR(host_buffer.input_grad[0], expected_input_grad, 1e-10)
        << "Backward pass failed: expected " << expected_input_grad 
        << " but got " << host_buffer.input_grad[0];
}

// Test with zero input value
TEST_F(BroadcastOperatorTest, BroadcastZeroValue) {
    BroadcastTestBuffer host_buffer;
    
    // Initialize input: zero value
    host_buffer.input_data[0] = 0.0f;
    host_buffer.input_grad[0] = 0.0f;
    
    // Set non-zero output gradients
    for (std::size_t i = 0; i < 4; ++i) {
        host_buffer.final_grad[i] = static_cast<float>(i + 1); // [1, 2, 3, 4]
    }
    
    // Copy to device and run
    auto device_buffer = makeCudaUnique<BroadcastTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(BroadcastTestBuffer), cudaMemcpyHostToDevice);
    
    test_broadcast_1_to_4_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(BroadcastTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify forward pass: all output elements should be zero
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(host_buffer.output_data[i], 0.0f, 1e-6f) 
            << "Forward pass failed at index " << i;
    }
    
    // Verify backward pass: input gradient should still be sum of output gradients
    float expected_input_grad = 1.0f + 2.0f + 3.0f + 4.0f; // = 10.0f
    EXPECT_NEAR(host_buffer.input_grad[0], expected_input_grad, 1e-6f)
        << "Backward pass failed with zero input value";
}

// Test chained operations with broadcast
struct ChainedBroadcastTestBuffer {
    float scalar_data[1];
    float scalar_grad[1];
    float vector_data[3];
    float vector_grad[3];
    float result_data[3];
    float result_grad[3];
    float final_grad[3];
};

__global__ void test_chained_broadcast_kernel(ChainedBroadcastTestBuffer* buffer) {
    VariableRef<1, float> scalar(buffer->scalar_data, buffer->scalar_grad);
    VariableRef<3, float> vector(buffer->vector_data, buffer->vector_grad);
    
    scalar.zero_grad();
    vector.zero_grad();
    
    // Chain: broadcast scalar to 3D, then add with vector
    auto broadcasted_scalar = op::broadcast<3>(scalar);
    auto result = broadcasted_scalar + vector;  // Using universal operators
    
    result.forward();
    
    // Copy result
    for (std::size_t i = 0; i < 3; ++i) {
        buffer->result_data[i] = result[i];
    }
    
    // Set gradients and backward
    for (std::size_t i = 0; i < 3; ++i) {
        result.add_grad(i, buffer->final_grad[i]);
    }
    result.backward();
}

TEST_F(BroadcastOperatorTest, ChainedBroadcastWithAddition) {
    ChainedBroadcastTestBuffer host_buffer;
    
    // Initialize: scalar = 2.0, vector = [1.0, 3.0, 5.0]
    host_buffer.scalar_data[0] = 2.0f;
    host_buffer.scalar_grad[0] = 0.0f;
    
    host_buffer.vector_data[0] = 1.0f;
    host_buffer.vector_data[1] = 3.0f;
    host_buffer.vector_data[2] = 5.0f;
    for (std::size_t i = 0; i < 3; ++i) {
        host_buffer.vector_grad[i] = 0.0f;
        host_buffer.final_grad[i] = 1.0f; // All output gradients = 1
    }
    
    // Copy to device and run
    auto device_buffer = makeCudaUnique<ChainedBroadcastTestBuffer>();
    cudaMemcpy(device_buffer.get(), &host_buffer, sizeof(ChainedBroadcastTestBuffer), cudaMemcpyHostToDevice);
    
    test_chained_broadcast_kernel<<<1, 1>>>(device_buffer.get());
    cudaDeviceSynchronize();
    
    cudaMemcpy(&host_buffer, device_buffer.get(), sizeof(ChainedBroadcastTestBuffer), cudaMemcpyDeviceToHost);
    
    // Verify forward pass: broadcast(2.0) + [1.0, 3.0, 5.0] = [3.0, 5.0, 7.0]
    float expected_results[3] = {3.0f, 5.0f, 7.0f};
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(host_buffer.result_data[i], expected_results[i], 1e-6f)
            << "Chained operation forward pass failed at index " << i;
    }
    
    // Verify backward pass:
    // Scalar gradient should be sum of all output gradients = 3.0
    EXPECT_NEAR(host_buffer.scalar_grad[0], 3.0f, 1e-6f)
        << "Scalar gradient incorrect in chained operation";
    
    // Vector gradients should each be 1.0 (direct passthrough from output)
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(host_buffer.vector_grad[i], 1.0f, 1e-6f)
            << "Vector gradient incorrect at index " << i;
    }
}