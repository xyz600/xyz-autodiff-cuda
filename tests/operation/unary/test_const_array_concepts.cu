#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../../../include/operations/unary/const_array_concepts.cuh"
#include "../../../include/variable.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;
using namespace xyz_autodiff::op;

static_assert(ArrayLikeConcept<Variable<3, float>>, "Variable should satisfy ArrayLikeConcept");
static_assert(ArrayLikeConcept<VariableRef<3, float>>, "VariableRef should satisfy ArrayLikeConcept");

class ConstArrayConceptsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

__global__ void test_array_like_concept_kernel(float* result) {
    Variable<3, float> var;
    var[0] = 1.0f;
    var[1] = 2.0f; 
    var[2] = 3.0f;
    
    if constexpr (ArrayLikeConcept<Variable<3, float>>) {
        result[0] = var[0] + var[1] + var[2];
    } else {
        result[0] = -1.0f;
    }
}

TEST_F(ConstArrayConceptsTest, ArrayLikeConceptWorksInKernel) {
    auto device_result = makeCudaUnique<float>();
    
    test_array_like_concept_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_result, 6.0f);
}