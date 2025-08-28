#include <gtest/gtest.h>
#include "../include/const_array.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

class ConstArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

__global__ void test_const_array_kernel(float* result) {
    ConstArray<float, 3> arr;
    arr[0] = 1.0f;
    arr[1] = 2.0f;
    arr[2] = 3.0f;
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
    result[3] = static_cast<float>(arr.size());
}

__global__ void test_const_array_initialization_kernel(float* result) {
    const float init_data[3] = {10.0f, 20.0f, 30.0f};
    ConstArray<float, 3> arr(init_data);
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
}

__global__ void test_const_array_const_access_kernel(float* result) {
    const float init_data[3] = {100.0f, 200.0f, 300.0f};
    const ConstArray<float, 3> arr(init_data);
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
}

TEST_F(ConstArrayTest, BasicOperations) {
    struct TestBuffers {
        float result[4];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 3.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[3], 3.0f);
}

TEST_F(ConstArrayTest, InitializationConstructor) {
    struct TestBuffers {
        float result[3];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_initialization_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 30.0f);
}

TEST_F(ConstArrayTest, ConstAccess) {
    struct TestBuffers {
        float result[3];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_const_access_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 100.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 200.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 300.0f);
}

TEST_F(ConstArrayTest, HostOperations) {
    ConstArray<int, 4> arr;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
    EXPECT_EQ(arr[3], 40);
    EXPECT_EQ(arr.size(), 4);
}

TEST_F(ConstArrayTest, HostInitialization) {
    const int init_data[5] = {1, 2, 3, 4, 5};
    ConstArray<int, 5> arr(init_data);
    
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    EXPECT_EQ(arr[4], 5);
    EXPECT_EQ(arr.size(), 5);
}