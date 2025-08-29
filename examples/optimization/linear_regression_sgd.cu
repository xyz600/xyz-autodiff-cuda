#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/operations/unary/squared_logic.cuh>

using namespace xyz_autodiff;

// 真の関数パラメータ
constexpr double TRUE_A = 2.5;
constexpr double TRUE_B = 1.8;
constexpr double TRUE_C = -1.2;
constexpr double TRUE_D = 0.7;

// ハイパーパラメータ
constexpr int TOTAL_SAMPLES = 100000;
constexpr int BATCH_SIZE = 256 * 32;
constexpr int NUM_EPOCHS = 10000;
constexpr double INITIAL_LR = 0.0001;
constexpr double FINAL_LR = INITIAL_LR / 100.0;
constexpr double NOISE_LEVEL = 0.5;

// データ生成用の構造体
struct DataPoint {
    double x1, x2, y;
};

// パラメータ管理用の構造体
struct Parameters {
    double value[4];  // a, b, c, d
    double grad[4];   // grad_a, grad_b, grad_c, grad_d
};

// ホストでデータ生成
std::vector<DataPoint> generate_data() {
    std::vector<DataPoint> data(TOTAL_SAMPLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> x_dist(-5.0, 5.0);
    std::normal_distribution<double> noise_dist(0.0, NOISE_LEVEL);
    
    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        double x1 = x_dist(gen);
        double x2 = x_dist(gen);
        double noise = noise_dist(gen);
        
        // y = (x1 - a)^2 + b(x2 - c)^2 + d + noise
        double y_true = (x1 - TRUE_A) * (x1 - TRUE_A) + 
                       TRUE_B * (x2 - TRUE_C) * (x2 - TRUE_C) + TRUE_D;
        
        data[i] = {x1, x2, y_true + noise};
    }
    
    std::cout << "Generated " << TOTAL_SAMPLES << " data points with noise level " << NOISE_LEVEL << std::endl;
    std::cout << "True parameters: a=" << TRUE_A << ", b=" << TRUE_B << ", c=" << TRUE_C << ", d=" << TRUE_D << std::endl;
    
    return data;
}

// バッチ選択用のCUDAカーネル
__global__ void select_batch_kernel(const DataPoint* all_data, DataPoint* batch_data,
                                   int* indices, int total_samples, int batch_size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // 各スレッドでcurandを初期化
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    
    // ランダムインデックス生成
    int random_idx = curand(&state) % total_samples;
    indices[idx] = random_idx;
    batch_data[idx] = all_data[random_idx];
}

// 勾配初期化はcudaMemsetで実行

// 並列SGD勾配計算カーネル（256x64 grid/block構成）
__global__ void parallel_gradient_computation_kernel(
    const DataPoint* batch_data, int batch_size, Parameters* params) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // パラメータ変数の作成（ポインタ演算で各パラメータを指定）
    VariableRef<1, double> a_var(&params->value[0], &params->grad[0]);  // a
    VariableRef<1, double> b_var(&params->value[1], &params->grad[1]);  // b
    VariableRef<1, double> c_var(&params->value[2], &params->grad[2]);  // c
    VariableRef<1, double> d_var(&params->value[3], &params->grad[3]);  // d
    
    // 現在のスレッドが担当するデータポイント
    const DataPoint& data = batch_data[idx];
    
    // (x1 - a)^2 をカスタムオペレーションで計算
    auto x1_term = a_var - data.x1;
    auto x1_term2 = op::squared(x1_term);
    
    // (x2 - c)^2 をカスタムオペレーションで計算
    auto x2_c = c_var - data.x2;
    auto x2_squared = op::squared(x2_c);
    
    // b * (x2 - c)^2
    auto x2_term = b_var * x2_squared;
    
    // (x1 - a)^2 + b * (x2 - c)^2
    auto combined_terms = x1_term + x2_term;
    
    // y_pred = (x1 - a)^2 + b * (x2 - c)^2 + d
    auto y_pred = combined_terms + d_var;
    
    // loss = (y_pred - y_target)^2 もカスタムオペレーションで計算
    auto loss = y_pred - data.y;
    auto loss_2 = op::squared(loss);
    
    loss.run();
}

// パラメータ更新カーネル
__global__ void update_parameters_kernel(Parameters* params, double learning_rate, int batch_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 勾配の平均化とパラメータ更新
        for (int i = 0; i < 4; ++i) {
            double avg_grad = params->grad[i] / batch_size;
            params->value[i] -= learning_rate * avg_grad;
        }
    }
}

// 学習率計算（指数減衰）
double compute_learning_rate(int epoch) {
    double decay_rate = log(FINAL_LR / INITIAL_LR) / NUM_EPOCHS;
    return INITIAL_LR * exp(decay_rate * epoch);
}

// パラメータの誤差を計算
void print_parameter_error(double a, double b, double c, double d, int epoch) {
    double error_a = abs(a - TRUE_A);
    double error_b = abs(b - TRUE_B);
    double error_c = abs(c - TRUE_C);
    double error_d = abs(d - TRUE_D);
    double total_error = error_a + error_b + error_c + error_d;
    
    printf("Epoch %d: a=%.4f(err:%.4f), b=%.4f(err:%.4f), c=%.4f(err:%.4f), d=%.4f(err:%.4f), total_err=%.4f\n",
           epoch, a, error_a, b, error_b, c, error_c, d, error_d, total_error);
}

int main() {
    // CUDAデバイス設定
    cudaSetDevice(0);
    
    // データ生成
    auto data = generate_data();
    
    // デバイスメモリ確保
    auto device_data = makeCudaUniqueArray<DataPoint>(TOTAL_SAMPLES);
    auto device_batch = makeCudaUniqueArray<DataPoint>(BATCH_SIZE);
    auto device_indices = makeCudaUniqueArray<int>(BATCH_SIZE);
    auto device_params = makeCudaUnique<Parameters>();
    
    // データをデバイスにコピー
    cudaMemcpy(device_data.get(), data.data(), TOTAL_SAMPLES * sizeof(DataPoint), cudaMemcpyHostToDevice);
    
    // パラメータの初期化
    Parameters init_params = {
        {0.0, 1.0, 0.0, 0.0},  // value: a, b, c, d
        {0.0, 0.0, 0.0, 0.0}   // grad: grad_a, grad_b, grad_c, grad_d
    };
    cudaMemcpy(device_params.get(), &init_params, sizeof(Parameters), cudaMemcpyHostToDevice);
    
    std::cout << "\nStarting optimization..." << std::endl;
    std::cout << "Initial parameters: a=" << init_params.value[0] << ", b=" << init_params.value[1] 
              << ", c=" << init_params.value[2] << ", d=" << init_params.value[3] << std::endl;
    
    // 初期誤差を表示
    print_parameter_error(init_params.value[0], init_params.value[1], init_params.value[2], init_params.value[3], 0);
    
    // 最適化ループ
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // 学習率計算
        double lr = compute_learning_rate(epoch);
        
        // バッチ選択
        unsigned long long seed = epoch * 12345ULL;
        dim3 batch_block_size(256);
        dim3 batch_grid_size((BATCH_SIZE + batch_block_size.x - 1) / batch_block_size.x);
        
        select_batch_kernel<<<batch_grid_size, batch_block_size>>>(
            device_data.get(), device_batch.get(), device_indices.get(),
            TOTAL_SAMPLES, BATCH_SIZE, seed);
        
        cudaDeviceSynchronize();
        
        // 勾配初期化（cudaMemsetを使用）
        cudaMemset(&(device_params.get()->grad), 0, sizeof(double) * 4);
        
        // デバッグ用：1つのスレッドで実行
        parallel_gradient_computation_kernel<<<1, 1>>>(
            device_batch.get(), 1, device_params.get());  // バッチサイズを1に制限
        
        // パラメータ更新
        update_parameters_kernel<<<1, 1>>>(device_params.get(), lr, 1);
        
        cudaDeviceSynchronize();
        
        // エポックごとに進捗表示（デバッグ用）
        if ((epoch + 1) % 100 == 0) {
            Parameters host_params;
            cudaMemcpy(&host_params, device_params.get(), sizeof(Parameters), cudaMemcpyDeviceToHost);
            
            printf("Epoch %d: LR=%.6f, grads=[%.6f,%.6f,%.6f,%.6f] - ", epoch + 1, lr, 
                   host_params.grad[0], host_params.grad[1], host_params.grad[2], host_params.grad[3]);
            print_parameter_error(host_params.value[0], host_params.value[1], 
                                host_params.value[2], host_params.value[3], epoch + 1);
            
            // NaNチェック
            if (isnan(host_params.value[0]) || isnan(host_params.value[1]) || 
                isnan(host_params.value[2]) || isnan(host_params.value[3])) {
                printf("ERROR: NaN detected at epoch %d, stopping...\n", epoch + 1);
                break;
            }
        }
    }
    
    // 最終結果表示
    Parameters final_params;
    cudaMemcpy(&final_params, device_params.get(), sizeof(Parameters), cudaMemcpyDeviceToHost);
    
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "True parameters:  a=" << TRUE_A << ", b=" << TRUE_B << ", c=" << TRUE_C << ", d=" << TRUE_D << std::endl;
    std::cout << "Final parameters: a=" << final_params.value[0] << ", b=" << final_params.value[1] 
              << ", c=" << final_params.value[2] << ", d=" << final_params.value[3] << std::endl;
    print_parameter_error(final_params.value[0], final_params.value[1], 
                         final_params.value[2], final_params.value[3], NUM_EPOCHS);
    
    return 0;
}