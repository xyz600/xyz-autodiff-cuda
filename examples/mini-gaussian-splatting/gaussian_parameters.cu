#include "gaussian_parameters.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

GaussianCollection::GaussianCollection() {
    // Allocate host memory
    host_params.resize(NUM_GAUSSIANS);
    host_grads.resize(NUM_GAUSSIANS);
    host_adam.resize(NUM_GAUSSIANS);
    
    // Allocate device memory using CUDA unique pointers
    device_params = makeCudaUniqueArray<GaussianParams>(NUM_GAUSSIANS);
    device_grads = makeCudaUniqueArray<GaussianGrads>(NUM_GAUSSIANS);
    device_adam = makeCudaUniqueArray<AdamState>(NUM_GAUSSIANS);
    
    // Initialize Adam state to zero
    for (int i = 0; i < NUM_GAUSSIANS; i++) {
        AdamState& adam = host_adam[i];
        memset(&adam, 0, sizeof(AdamState));
    }
}

void GaussianCollection::initialize_random(int image_width, int image_height, std::mt19937& rng) {
    int max_size = std::max(image_width, image_height);
    float std_dev = static_cast<float>(max_size) / 30.0f;
    
    std::uniform_real_distribution<float> pos_x_dist(0.0f, static_cast<float>(image_width));
    std::uniform_real_distribution<float> pos_y_dist(0.0f, static_cast<float>(image_height));
    std::uniform_real_distribution<float> rotation_dist(-3.14159f, 3.14159f);
    std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> opacity_dist(0.1f, 0.3f);
    std::normal_distribution<float> scale_dist(std_dev, std_dev * 0.1f);  // Some variation around base scale
    
    std::cout << "Initializing " << NUM_GAUSSIANS << " Gaussians..." << std::endl;
    std::cout << "Image size: " << image_width << "x" << image_height << std::endl;
    std::cout << "Base standard deviation: " << std_dev << std::endl;
    
    for (int i = 0; i < NUM_GAUSSIANS; i++) {
        GaussianParams& params = host_params[i];
        
        // Random center position
        params.center[0] = pos_x_dist(rng);
        params.center[1] = pos_y_dist(rng);
        
        // Scale with some variation, but ensure positive values
        params.scale[0] = std::max(1.0f, scale_dist(rng));
        params.scale[1] = std::max(1.0f, scale_dist(rng));
        
        // Random rotation
        params.rotation[0] = rotation_dist(rng);
        
        // Random color
        params.color[0] = color_dist(rng);
        params.color[1] = color_dist(rng);
        params.color[2] = color_dist(rng);
        
        // Random opacity
        params.opacity[0] = opacity_dist(rng);
    }
    
    // Clear gradients
    zero_gradients();
    
    std::cout << "Gaussian initialization complete." << std::endl;
}

void GaussianCollection::upload_to_device() {
    cudaError_t err;
    
    err = cudaMemcpy(device_params.get(), host_params.data(), 
                     NUM_GAUSSIANS * sizeof(GaussianParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload Gaussian parameters: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(device_grads.get(), host_grads.data(),
                     NUM_GAUSSIANS * sizeof(GaussianGrads), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload gradients: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(device_adam.get(), host_adam.data(),
                     NUM_GAUSSIANS * sizeof(AdamState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload Adam state: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void GaussianCollection::download_from_device() {
    cudaError_t err;
    
    err = cudaMemcpy(host_params.data(), device_params.get(),
                     NUM_GAUSSIANS * sizeof(GaussianParams), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download Gaussian parameters: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(host_grads.data(), device_grads.get(),
                     NUM_GAUSSIANS * sizeof(GaussianGrads), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download gradients: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(host_adam.data(), device_adam.get(),
                     NUM_GAUSSIANS * sizeof(AdamState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download Adam state: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void GaussianCollection::zero_gradients() {
    for (int i = 0; i < NUM_GAUSSIANS; i++) {
        GaussianGrads& grad = host_grads[i];
        memset(&grad, 0, sizeof(GaussianGrads));
    }
}

void GaussianCollection::adam_step(float learning_rate, float beta1, float beta2, 
                                   float epsilon, int iteration) {
    float beta1_t = std::pow(beta1, iteration);
    float beta2_t = std::pow(beta2, iteration);
    float lr_corrected = learning_rate * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int i = 0; i < NUM_GAUSSIANS; i++) {
        GaussianParams& params = host_params[i];
        GaussianGrads& grads = host_grads[i];
        AdamState& adam = host_adam[i];
        
        // Update center
        for (int j = 0; j < 2; j++) {
            adam.m_center[j] = beta1 * adam.m_center[j] + (1.0f - beta1) * grads.center[j];
            adam.v_center[j] = beta2 * adam.v_center[j] + (1.0f - beta2) * grads.center[j] * grads.center[j];
            params.center[j] -= lr_corrected * adam.m_center[j] / (std::sqrt(adam.v_center[j]) + epsilon);
        }
        
        // Update scale (ensure positive)
        for (int j = 0; j < 2; j++) {
            adam.m_scale[j] = beta1 * adam.m_scale[j] + (1.0f - beta1) * grads.scale[j];
            adam.v_scale[j] = beta2 * adam.v_scale[j] + (1.0f - beta2) * grads.scale[j] * grads.scale[j];
            params.scale[j] -= lr_corrected * adam.m_scale[j] / (std::sqrt(adam.v_scale[j]) + epsilon);
            params.scale[j] = std::max(0.1f, params.scale[j]);  // Keep scale positive
        }
        
        // Update rotation
        adam.m_rotation[0] = beta1 * adam.m_rotation[0] + (1.0f - beta1) * grads.rotation[0];
        adam.v_rotation[0] = beta2 * adam.v_rotation[0] + (1.0f - beta2) * grads.rotation[0] * grads.rotation[0];
        params.rotation[0] -= lr_corrected * adam.m_rotation[0] / (std::sqrt(adam.v_rotation[0]) + epsilon);
        
        // Update color (clamp to [0, 1])
        for (int j = 0; j < 3; j++) {
            adam.m_color[j] = beta1 * adam.m_color[j] + (1.0f - beta1) * grads.color[j];
            adam.v_color[j] = beta2 * adam.v_color[j] + (1.0f - beta2) * grads.color[j] * grads.color[j];
            params.color[j] -= lr_corrected * adam.m_color[j] / (std::sqrt(adam.v_color[j]) + epsilon);
            params.color[j] = std::max(0.0f, std::min(1.0f, params.color[j]));
        }
        
        // Update opacity (clamp to [0, 1])
        adam.m_opacity[0] = beta1 * adam.m_opacity[0] + (1.0f - beta1) * grads.opacity[0];
        adam.v_opacity[0] = beta2 * adam.v_opacity[0] + (1.0f - beta2) * grads.opacity[0] * grads.opacity[0];
        params.opacity[0] -= lr_corrected * adam.m_opacity[0] / (std::sqrt(adam.v_opacity[0]) + epsilon);
        params.opacity[0] = std::max(0.01f, std::min(1.0f, params.opacity[0]));
    }
}

// CUDA kernel for Adam optimization step
__global__ void adam_step_kernel(
    GaussianParams* params,
    GaussianGrads* grads,
    AdamState* adam_states,
    int num_gaussians,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,
    float beta2_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float lr_corrected = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    GaussianParams& param = params[idx];
    GaussianGrads& grad = grads[idx];
    AdamState& adam = adam_states[idx];
    
    // Update center
    for (int j = 0; j < 2; j++) {
        adam.m_center[j] = beta1 * adam.m_center[j] + (1.0f - beta1) * grad.center[j];
        adam.v_center[j] = beta2 * adam.v_center[j] + (1.0f - beta2) * grad.center[j] * grad.center[j];
        param.center[j] -= lr_corrected * adam.m_center[j] / (sqrtf(adam.v_center[j]) + epsilon);
    }
    
    // Update scale (ensure positive)
    for (int j = 0; j < 2; j++) {
        adam.m_scale[j] = beta1 * adam.m_scale[j] + (1.0f - beta1) * grad.scale[j];
        adam.v_scale[j] = beta2 * adam.v_scale[j] + (1.0f - beta2) * grad.scale[j] * grad.scale[j];
        param.scale[j] -= lr_corrected * adam.m_scale[j] / (sqrtf(adam.v_scale[j]) + epsilon);
        param.scale[j] = fmaxf(0.1f, param.scale[j]);  // Keep scale positive
    }
    
    // Update rotation
    adam.m_rotation[0] = beta1 * adam.m_rotation[0] + (1.0f - beta1) * grad.rotation[0];
    adam.v_rotation[0] = beta2 * adam.v_rotation[0] + (1.0f - beta2) * grad.rotation[0] * grad.rotation[0];
    param.rotation[0] -= lr_corrected * adam.m_rotation[0] / (sqrtf(adam.v_rotation[0]) + epsilon);
    
    // Update color (clamp to [0, 1])
    for (int j = 0; j < 3; j++) {
        adam.m_color[j] = beta1 * adam.m_color[j] + (1.0f - beta1) * grad.color[j];
        adam.v_color[j] = beta2 * adam.v_color[j] + (1.0f - beta2) * grad.color[j] * grad.color[j];
        param.color[j] -= lr_corrected * adam.m_color[j] / (sqrtf(adam.v_color[j]) + epsilon);
        param.color[j] = fmaxf(0.0f, fminf(1.0f, param.color[j]));
    }
    
    // Update opacity (clamp to [0.01, 1])
    adam.m_opacity[0] = beta1 * adam.m_opacity[0] + (1.0f - beta1) * grad.opacity[0];
    adam.v_opacity[0] = beta2 * adam.v_opacity[0] + (1.0f - beta2) * grad.opacity[0] * grad.opacity[0];
    param.opacity[0] -= lr_corrected * adam.m_opacity[0] / (sqrtf(adam.v_opacity[0]) + epsilon);
    param.opacity[0] = fmaxf(0.01f, fminf(1.0f, param.opacity[0]));
}

void GaussianCollection::adam_step_gpu(float learning_rate, float beta1, float beta2, 
                                       float epsilon, int iteration) {
    float beta1_t = powf(beta1, iteration);
    float beta2_t = powf(beta2, iteration);
    
    // Calculate grid dimensions
    int block_size = 256;
    int grid_size = (NUM_GAUSSIANS + block_size - 1) / block_size;
    
    // Launch kernel
    adam_step_kernel<<<grid_size, block_size>>>(
        device_params.get(),
        device_grads.get(),
        device_adam.get(),
        NUM_GAUSSIANS,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        beta1_t,
        beta2_t
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Adam kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Adam kernel execution error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}