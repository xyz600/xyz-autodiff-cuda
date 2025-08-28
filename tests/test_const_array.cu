#include <gtest/gtest.h>
#include "../include/const_array.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

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

__global__ void test_const_array_operators_kernel(float* result) {
    const float init_data1[3] = {10.0f, 20.0f, 30.0f};
    const float init_data2[3] = {5.0f, 15.0f, 25.0f};
    
    ConstArray<float, 3> arr1(init_data1);
    ConstArray<float, 3> arr2(init_data2);
    
    // Test assignment operator
    ConstArray<float, 3> arr_copy;
    arr_copy = arr1;  // Copy arr1 to arr_copy
    result[0] = arr_copy[0];  // Should be 10.0f
    result[1] = arr_copy[1];  // Should be 20.0f
    result[2] = arr_copy[2];  // Should be 30.0f
    
    // Test compound assignment
    arr1 += arr2;  // arr1 becomes [15, 35, 55]
    result[3] = arr1[0];
    result[4] = arr1[1];
    result[5] = arr1[2];
    
    // Test binary subtraction
    auto arr3 = arr1 - arr2;  // [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    result[6] = arr3[0];
    result[7] = arr3[1];
    result[8] = arr3[2];
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

TEST_F(ConstArrayTest, ArithmeticOperators) {
    struct TestBuffers {
        float result[9];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_operators_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    // Test assignment results: arr_copy = arr1 (original [10, 20, 30])
    EXPECT_FLOAT_EQ(host_buffers.result[0], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 30.0f);
    
    // Test compound assignment results: [10, 20, 30] += [5, 15, 25] = [15, 35, 55]
    EXPECT_FLOAT_EQ(host_buffers.result[3], 15.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[4], 35.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[5], 55.0f);
    
    // Test binary subtraction results: [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    EXPECT_FLOAT_EQ(host_buffers.result[6], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[7], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[8], 30.0f);
}

TEST_F(ConstArrayTest, HostArithmeticOperators) {
    ConstArray<float, 3> arr1;
    arr1[0] = 10.0f; arr1[1] = 20.0f; arr1[2] = 30.0f;
    
    ConstArray<float, 3> arr2;
    arr2[0] = 5.0f; arr2[1] = 15.0f; arr2[2] = 25.0f;
    
    // Test assignment operator
    ConstArray<float, 3> arr_copy;
    arr_copy = arr1;
    EXPECT_FLOAT_EQ(arr_copy[0], 10.0f);
    EXPECT_FLOAT_EQ(arr_copy[1], 20.0f);
    EXPECT_FLOAT_EQ(arr_copy[2], 30.0f);
    
    // Test compound assignment
    arr1 += arr2;
    EXPECT_FLOAT_EQ(arr1[0], 15.0f);
    EXPECT_FLOAT_EQ(arr1[1], 35.0f);
    EXPECT_FLOAT_EQ(arr1[2], 55.0f);
    
    // Test binary operations
    auto diff = arr1 - arr2;  // [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    EXPECT_FLOAT_EQ(diff[0], 10.0f);
    EXPECT_FLOAT_EQ(diff[1], 20.0f);
    EXPECT_FLOAT_EQ(diff[2], 30.0f);
    
    auto sum = arr1 + arr2;  // [15, 35, 55] + [5, 15, 25] = [20, 50, 80]
    EXPECT_FLOAT_EQ(sum[0], 20.0f);
    EXPECT_FLOAT_EQ(sum[1], 50.0f);
    EXPECT_FLOAT_EQ(sum[2], 80.0f);
}