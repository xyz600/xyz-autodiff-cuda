#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

// テスト用バッファ構造体
struct SharedMemoryTestBuffers {
    double x_grad;
    double y_grad;
};

// マルチブロック用バッファ構造体
struct MultiBlockTestBuffers {
    double x_grads[10];  // 最大10ブロック分
    double y_grads[10];  // 最大10ブロック分
};

/**
 * Shared Memory AtomicAdd テスト
 * shared memory上の変数に対して複数スレッドが
 * atomicAddで勾配を加算することを確認
 */

__global__ void shared_memory_atomic_kernel(
    SharedMemoryTestBuffers* global_results,
    std::size_t threads_per_block) {
    
    // Shared memory allocation
    __shared__ double shared_x_data[1];
    __shared__ double shared_x_grad[1];
    __shared__ double shared_y_data[1];
    __shared__ double shared_y_grad[1];
    
    // Initialize shared memory (only thread 0)
    if (threadIdx.x == 0) {
        shared_x_data[0] = 5.0;
        shared_x_grad[0] = 0.0;
        shared_y_data[0] = 3.0;
        shared_y_grad[0] = 0.0;
    }
    
    __syncthreads(); // Ensure initialization is complete
    
    // Create VariableRef pointing to shared memory
    VariableRef<1, double> x_ref(shared_x_data, shared_x_grad);
    VariableRef<1, double> y_ref(shared_y_data, shared_y_grad);
    
    // Each thread adds gradient using atomicAdd on shared memory
    x_ref.add_grad(0, 1.0);  // atomicAdd on shared memory
    y_ref.add_grad(0, 2.0);  // atomicAdd on shared memory
    
    __syncthreads(); // Ensure all atomic operations are complete
    
    // Copy results back to global memory (only thread 0)
    if (threadIdx.x == 0) {
        global_results->x_grad = shared_x_grad[0];
        global_results->y_grad = shared_y_grad[0];
    }
}

class SharedMemoryAtomicTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(SharedMemoryAtomicTest, SharedMemoryAtomicAddGradient) {
    const std::size_t THREADS_PER_BLOCK = 128;
    const std::size_t NUM_BLOCKS = 1; // Single block to test shared memory
    
    // Device memory for results
    auto device_results = makeCudaUnique<SharedMemoryTestBuffers>();
    ASSERT_NE(device_results, nullptr);
    
    // Initialize results to zero
    SharedMemoryTestBuffers zero_buffers = {0.0, 0.0};
    ASSERT_EQ(cudaMemcpy(device_results.get(), &zero_buffers, sizeof(SharedMemoryTestBuffers), cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    shared_memory_atomic_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        device_results.get(),
        THREADS_PER_BLOCK
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy results back to host
    SharedMemoryTestBuffers host_results;
    ASSERT_EQ(cudaMemcpy(&host_results, device_results.get(), sizeof(SharedMemoryTestBuffers), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Verify results
    // Each of THREADS_PER_BLOCK threads adds 1.0 to x_grad
    double expected_x_grad = static_cast<double>(THREADS_PER_BLOCK);
    // Each of THREADS_PER_BLOCK threads adds 2.0 to y_grad
    double expected_y_grad = static_cast<double>(THREADS_PER_BLOCK) * 2.0;
    
    EXPECT_NEAR(host_results.x_grad, expected_x_grad, 1e-10)
        << "x gradient should be " << expected_x_grad << " but got " << host_results.x_grad;
    EXPECT_NEAR(host_results.y_grad, expected_y_grad, 1e-10)
        << "y gradient should be " << expected_y_grad << " but got " << host_results.y_grad;
    
    // Success message
    std::cout << "SUCCESS: Shared memory atomicAdd works correctly!" << std::endl;
    std::cout << "x_grad: " << host_results.x_grad << " (expected: " << expected_x_grad << ")" << std::endl;
    std::cout << "y_grad: " << host_results.y_grad << " (expected: " << expected_y_grad << ")" << std::endl;
    std::cout << "AtomicAdd performed correctly on shared memory with " << THREADS_PER_BLOCK << " threads." << std::endl;
}

// Multi-block shared memory atomic kernel
__global__ void multi_block_shared_memory_kernel(MultiBlockTestBuffers* results, std::size_t threads_per_block) {
    // Shared memory allocation (per block)
    __shared__ double shared_x_data[1];
    __shared__ double shared_x_grad[1];
    __shared__ double shared_y_data[1];
    __shared__ double shared_y_grad[1];
    
    // Initialize shared memory (only thread 0 in each block)
    if (threadIdx.x == 0) {
        shared_x_data[0] = 5.0;
        shared_x_grad[0] = 0.0;
        shared_y_data[0] = 3.0;
        shared_y_grad[0] = 0.0;
    }
    
    __syncthreads();
    
    // Create VariableRef pointing to shared memory
    VariableRef<1, double> x_ref(shared_x_data, shared_x_grad);
    VariableRef<1, double> y_ref(shared_y_data, shared_y_grad);
    
    // Each thread adds gradient using atomicAdd on shared memory
    x_ref.add_grad(0, 1.0);
    y_ref.add_grad(0, 2.0);
    
    __syncthreads();
    
    // Copy results back to global memory (only thread 0 in each block)
    if (threadIdx.x == 0) {
        results->x_grads[blockIdx.x] = shared_x_grad[0];
        results->y_grads[blockIdx.x] = shared_y_grad[0];
    }
}

// Test with multiple blocks to verify shared memory isolation
TEST_F(SharedMemoryAtomicTest, MultiBlockSharedMemoryAtomic) {
    const std::size_t THREADS_PER_BLOCK = 64;
    const std::size_t NUM_BLOCKS = 4;
    
    // Device memory for results (one per block)
    auto device_results = makeCudaUnique<MultiBlockTestBuffers>();
    ASSERT_NE(device_results, nullptr);
    
    // Initialize all results to zero
    MultiBlockTestBuffers zero_results = {};
    ASSERT_EQ(cudaMemcpy(device_results.get(), &zero_results, sizeof(MultiBlockTestBuffers), cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    multi_block_shared_memory_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        device_results.get(),
        THREADS_PER_BLOCK
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy results back to host
    MultiBlockTestBuffers host_results;
    ASSERT_EQ(cudaMemcpy(&host_results, device_results.get(), sizeof(MultiBlockTestBuffers), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Verify results for each block
    double expected_x_grad = static_cast<double>(THREADS_PER_BLOCK);
    double expected_y_grad = static_cast<double>(THREADS_PER_BLOCK) * 2.0;
    
    for (std::size_t block = 0; block < NUM_BLOCKS; ++block) {
        EXPECT_NEAR(host_results.x_grads[block], expected_x_grad, 1e-10)
            << "Block " << block << " x gradient should be " << expected_x_grad << " but got " << host_results.x_grads[block];
        EXPECT_NEAR(host_results.y_grads[block], expected_y_grad, 1e-10)
            << "Block " << block << " y gradient should be " << expected_y_grad << " but got " << host_results.y_grads[block];
    }
    
    std::cout << "SUCCESS: Multi-block shared memory atomicAdd works correctly!" << std::endl;
    std::cout << "Tested " << NUM_BLOCKS << " blocks with " << THREADS_PER_BLOCK << " threads each." << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}