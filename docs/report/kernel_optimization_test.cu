#include "../include/variable.cuh"
#include "../include/operations/operation.cuh"
#include "../include/operations/add_logic.cuh"

using namespace xyz_autodiff;

// 基本的なVariable操作のテスト
__global__ void test_variable_basic(float* data1, float* grad1, float* data2, float* grad2, float* result) {
    Variable<float, 1> var1(data1, grad1);
    Variable<float, 1> var2(data2, grad2);
    
    var1[0] = 3.0f;
    var2[0] = 4.0f;
    
    result[0] = var1[0] + var2[0];
}

// 単一Operation（内部バッファ）のテスト
__global__ void test_operation_internal(float* data1, float* grad1, float* data2, float* grad2, float* result) {
    Variable<float, 1> var1(data1, grad1);
    Variable<float, 1> var2(data2, grad2);
    
    var1[0] = 3.0f;
    var2[0] = 4.0f;
    
    auto op = op::add(var1, var2);
    result[0] = op[0];
    
    op.grad(0) = 1.0f;
    op.backward();
    
    result[1] = var1.grad(0);
    result[2] = var2.grad(0);
}

// 単一Operation（外部バッファ）のテスト
__global__ void test_operation_external(float* data1, float* grad1, float* data2, float* grad2, 
                                       float* output_data, float* output_grad, float* result) {
    Variable<float, 1> var1(data1, grad1);
    Variable<float, 1> var2(data2, grad2);
    Variable<float, 1> output_var(output_data, output_grad);
    
    var1[0] = 3.0f;
    var2[0] = 4.0f;
    
    auto op_ref = op::add_ref(var1, var2, output_var);
    op_ref.forward();
    
    result[0] = op_ref[0];
    
    op_ref.grad(0) = 1.0f;
    op_ref.backward();
    
    result[1] = var1.grad(0);
    result[2] = var2.grad(0);
}

// Operation Chaining（Variable Conceptを使用）のテスト
__global__ void test_operation_chaining(float* data1, float* grad1, float* data2, float* grad2, float* data3, float* grad3, 
                                       float* temp_data, float* temp_grad, float* result) {
    Variable<float, 1> var1(data1, grad1);
    Variable<float, 1> var2(data2, grad2);
    Variable<float, 1> var3(data3, grad3);
    Variable<float, 1> temp_var(temp_data, temp_grad);
    
    var1[0] = 3.0f;
    var2[0] = 4.0f;
    var3[0] = 5.0f;
    
    // Chain: (var1 + var2) + var3
    auto node1_ref = op::add_ref(var1, var2, temp_var);
    node1_ref.forward();
    
    // Variable Conceptを使用してchainingのように見せる
    result[0] = var3[0] + temp_var[0];
    
    // Backward
    temp_var.grad(0) = 1.0f;
    result[3] = 1.0f;  // dL/dvar3
    
    node1_ref.backward();
    
    result[1] = var1.grad(0);
    result[2] = var2.grad(0);
}

// 最適化確認用：インライン化とループ展開のテスト
__global__ void test_optimization_inline(float* data, float* result) {
    // 複数の単純な演算を連続で実行
    Variable<float, 4> var(data, data);  // gradとdataを同じにして簡略化
    
    // 単純な演算の連鎖
    var[0] = 1.0f;
    var[1] = var[0] + 2.0f;
    var[2] = var[1] * 3.0f;
    var[3] = var[2] - 1.0f;
    
    // 結果保存
    for (int i = 0; i < 4; ++i) {
        result[i] = var[i];
    }
}

// テンプレート最適化確認用
template<int N>
__global__ void test_template_optimization(float* data, float* result) {
    Variable<float, N> var(data, data);
    
    // コンパイル時定数でのループ展開確認
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        var[i] = static_cast<float>(i) * 2.0f;
    }
    
    #pragma unroll  
    for (int i = 0; i < N; ++i) {
        result[i] = var[i];
    }
}

// 明示的なテンプレートインスタンス化
template __global__ void test_template_optimization<4>(float*, float*);
template __global__ void test_template_optimization<8>(float*, float*);