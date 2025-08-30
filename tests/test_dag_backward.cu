#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

// DAGテスト用のバッファ構造体
template<typename T>
struct DAGTestBuffers {
    T a_data[1];
    T a_grad[1];
    T b_data[1];
    T b_grad[1];
    T c_data[1];
    T c_grad[1];
    T result;
};

class DAGBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// DAGの例:
//     a
//    / \
//   b   c
//    \ /
//     d
// b = a + 2, c = a * 3, d = b + c
// a=1の時: b=3, c=3, d=6
// d/da = d/db * db/da + d/dc * dc/da = 1*1 + 1*3 = 4
__global__ void test_dag_single_shared_node_kernel(DAGTestBuffers<double>* buffers) {
    // 初期値設定
    buffers->a_data[0] = 1.0;
    buffers->a_grad[0] = 0.0;
    buffers->b_data[0] = 2.0;
    buffers->b_grad[0] = 0.0;
    buffers->c_data[0] = 3.0;
    buffers->c_grad[0] = 0.0;
    
    // 変数の作成
    VariableRef<1, double> a_var(buffers->a_data, buffers->a_grad);
    VariableRef<1, double> two_var(buffers->b_data, buffers->b_grad);
    VariableRef<1, double> three_var(buffers->c_data, buffers->c_grad);
    
    // DAGの構築: aが共有ノード
    auto b = op::add(a_var, two_var);     // b = a + 2
    auto c = op::mul(a_var, three_var);   // c = a * 3
    auto d = op::add(b, c);                // d = b + c
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    d.run();
    
    // 結果保存
    buffers->result = buffers->a_grad[0];
}

TEST_F(DAGBackwardTest, SingleSharedNode) {
    auto device_buffers = makeCudaUnique<DAGTestBuffers<double>>();
    
    test_dag_single_shared_node_kernel<<<1, 1>>>(device_buffers.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DAGTestBuffers<double> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
              sizeof(DAGTestBuffers<double>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // d/da = d/db * db/da + d/dc * dc/da = 1*1 + 1*3 = 4
    EXPECT_NEAR(host_buffers.result, 4.0, 1e-10);
}

// より複雑なDAGの例:
//     a
//    /|\
//   b c |
//    \|/
//     d
// b = a * 2, c = a + 1, d = b + c + a
// a=2の時: b=4, c=3, d=4+3+2=9
// d/da = d/db * db/da + d/dc * dc/da + d/da(直接) = 1*2 + 1*1 + 1 = 4
__global__ void test_dag_multiple_paths_kernel(DAGTestBuffers<double>* buffers) {
    // 初期値設定
    buffers->a_data[0] = 2.0;
    buffers->a_grad[0] = 0.0;
    buffers->b_data[0] = 2.0;  // 定数2
    buffers->b_grad[0] = 0.0;
    buffers->c_data[0] = 1.0;  // 定数1
    buffers->c_grad[0] = 0.0;
    
    // 変数の作成
    VariableRef<1, double> a_var(buffers->a_data, buffers->a_grad);
    VariableRef<1, double> two_var(buffers->b_data, buffers->b_grad);
    VariableRef<1, double> one_var(buffers->c_data, buffers->c_grad);
    
    // DAGの構築: aが3つのパスを持つ
    auto b = op::mul(a_var, two_var);     // b = a * 2
    auto c = op::add(a_var, one_var);     // c = a + 1
    auto temp = op::add(b, c);            // temp = b + c
    auto d = op::add(temp, a_var);        // d = temp + a
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    d.run();
    
    // 結果保存
    buffers->result = buffers->a_grad[0];
}

TEST_F(DAGBackwardTest, MultiplePathsToSameNode) {
    auto device_buffers = makeCudaUnique<DAGTestBuffers<double>>();
    
    test_dag_multiple_paths_kernel<<<1, 1>>>(device_buffers.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DAGTestBuffers<double> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
              sizeof(DAGTestBuffers<double>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // d/da = d/db * db/da + d/dc * dc/da + d/da(直接) = 1*2 + 1*1 + 1 = 4
    EXPECT_NEAR(host_buffers.result, 4.0, 1e-10);
}

// 同じ変数を複数回使用するケース:
//   a
//   |
//   b = a * a
//   |
//   c = b + a
// a=3の時: b=9, c=12
// d/da = dc/db * db/da + dc/da(直接) = 1 * 2a + 1 = 1*6 + 1 = 7
__global__ void test_dag_self_multiplication_kernel(DAGTestBuffers<double>* buffers) {
    // 初期値設定
    buffers->a_data[0] = 3.0;
    buffers->a_grad[0] = 0.0;
    
    // 変数の作成
    VariableRef<1, double> a_var(buffers->a_data, buffers->a_grad);
    
    // DAGの構築: a * a
    auto b = op::mul(a_var, a_var);       // b = a * a
    auto c = op::add(b, a_var);           // c = b + a
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    c.run();
    
    // 結果保存
    buffers->result = buffers->a_grad[0];
}

TEST_F(DAGBackwardTest, SelfMultiplication) {
    auto device_buffers = makeCudaUnique<DAGTestBuffers<double>>();
    
    test_dag_self_multiplication_kernel<<<1, 1>>>(device_buffers.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DAGTestBuffers<double> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
              sizeof(DAGTestBuffers<double>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // dc/da = dc/db * db/da + dc/da(直接) = 1 * 2a + 1 = 1*6 + 1 = 7
    EXPECT_NEAR(host_buffers.result, 7.0, 1e-10);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}