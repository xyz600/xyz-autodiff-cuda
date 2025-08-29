#include "gaussian_parameters.h"
#include "training_config.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

GaussianCollection::GaussianCollection(const TrainingConfig& config) : num_gaussians(config.num_gaussians) {
    // Allocate host memory
    host_params.resize(num_gaussians);
    host_grads.resize(num_gaussians);
    host_adam.resize(num_gaussians);
    
    // Allocate device memory using CUDA unique pointers
    device_params = makeCudaUniqueArray<GaussianParams>(num_gaussians);
    device_grads = makeCudaUniqueArray<GaussianGrads>(num_gaussians);
    device_adam = makeCudaUniqueArray<AdamState>(num_gaussians);
    
    // Initialize Adam state to zero
    for (int i = 0; i < num_gaussians; i++) {
        AdamState& adam = host_adam[i];
        memset(&adam, 0, sizeof(AdamState));
    }
}

void GaussianCollection::initialize_random(int image_width, int image_height, std::mt19937& rng) {
    int max_size = std::max(image_width, image_height);
    
    std::uniform_real_distribution<float> pos_x_dist(0.0f, static_cast<float>(image_width));
    std::uniform_real_distribution<float> pos_y_dist(0.0f, static_cast<float>(image_height));
    std::uniform_real_distribution<float> color_dist(0.1f, 0.2f);
    std::uniform_real_distribution<float> opacity_dist(0.05f, 0.1f);
    std::uniform_real_distribution<float> scale_dist(0.0f, 2.0f);  // Some variation around base scale
    
    std::cout << "Initializing " << num_gaussians << " Gaussians..." << std::endl;
    std::cout << "Image size: " << image_width << "x" << image_height << std::endl;
    
    for (int i = 0; i < num_gaussians; i++) {
        GaussianParams& params = host_params[i];
        
        // Random center position
        params.center[0] = pos_x_dist(rng);
        params.center[1] = pos_y_dist(rng);
        
        // Scale with some variation, but ensure positive values
        params.scale[0] = std::max(1.0f, scale_dist(rng));
        params.scale[1] = std::max(1.0f, scale_dist(rng));
        
        // Random rotation
        params.rotation[0] = 0.0f;
        
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
                     num_gaussians * sizeof(GaussianParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload Gaussian parameters: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(device_grads.get(), host_grads.data(),
                     num_gaussians * sizeof(GaussianGrads), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload gradients: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(device_adam.get(), host_adam.data(),
                     num_gaussians * sizeof(AdamState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to upload Adam state: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void GaussianCollection::download_from_device() {
    cudaError_t err;
    
    err = cudaMemcpy(host_params.data(), device_params.get(),
                     num_gaussians * sizeof(GaussianParams), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download Gaussian parameters: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(host_grads.data(), device_grads.get(),
                     num_gaussians * sizeof(GaussianGrads), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download gradients: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMemcpy(host_adam.data(), device_adam.get(),
                     num_gaussians * sizeof(AdamState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to download Adam state: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void GaussianCollection::zero_gradients() {
    for (int i = 0; i < num_gaussians; i++) {
        GaussianGrads& grad = host_grads[i];
        memset(&grad, 0, sizeof(GaussianGrads));
    }
}

void GaussianCollection::adam_step(float learning_rate, float beta1, float beta2, 
                                   float epsilon, int iteration) {
    float beta1_t = std::pow(beta1, iteration);
    float beta2_t = std::pow(beta2, iteration);
    float lr_corrected = learning_rate * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int i = 0; i < num_gaussians; i++) {
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
    
    // Update scale
    for (int j = 0; j < 2; j++) {
        adam.m_scale[j] = beta1 * adam.m_scale[j] + (1.0f - beta1) * grad.scale[j];
        adam.v_scale[j] = beta2 * adam.v_scale[j] + (1.0f - beta2) * grad.scale[j] * grad.scale[j];
        param.scale[j] -= lr_corrected * adam.m_scale[j] / (sqrtf(adam.v_scale[j]) + epsilon);
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
    }
    
    // Update opacity (clamp to [0.01, 1])
    adam.m_opacity[0] = beta1 * adam.m_opacity[0] + (1.0f - beta1) * grad.opacity[0];
    adam.v_opacity[0] = beta2 * adam.v_opacity[0] + (1.0f - beta2) * grad.opacity[0] * grad.opacity[0];
    param.opacity[0] -= lr_corrected * adam.m_opacity[0] / (sqrtf(adam.v_opacity[0]) + epsilon);
}

// CUDA kernel to zero gradients on device
__global__ void zero_gradients_kernel(GaussianGrads* grads, int num_gaussians) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    GaussianGrads& grad = grads[idx];
    grad.center[0] = 0.0f;
    grad.center[1] = 0.0f;
    grad.scale[0] = 0.0f;
    grad.scale[1] = 0.0f;
    grad.rotation[0] = 0.0f;
    grad.color[0] = 0.0f;
    grad.color[1] = 0.0f;
    grad.color[2] = 0.0f;
    grad.opacity[0] = 0.0f;
}

void GaussianCollection::zero_gradients_gpu() {
    // Calculate grid dimensions
    int block_size = 256;
    int grid_size = (num_gaussians + block_size - 1) / block_size;
    
    // Launch kernel to zero gradients on device
    zero_gradients_kernel<<<grid_size, block_size>>>(device_grads.get(), num_gaussians);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Zero gradients kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

// CUDA kernel for Adam optimization step with individual learning rates
__global__ void adam_step_individual_kernel(
    GaussianParams* params,
    GaussianGrads* grads,
    AdamState* adam_states,
    int num_gaussians,
    float lr_center,
    float lr_scale,
    float lr_rotation,
    float lr_color,
    float lr_opacity,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,
    float beta2_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    GaussianParams& param = params[idx];
    GaussianGrads& grad = grads[idx];
    AdamState& adam = adam_states[idx];
    
    // Compute corrected learning rates
    float lr_center_corrected = lr_center * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    float lr_scale_corrected = lr_scale * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    float lr_rotation_corrected = lr_rotation * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    float lr_color_corrected = lr_color * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    float lr_opacity_corrected = lr_opacity * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update center
    for (int j = 0; j < 2; j++) {
        adam.m_center[j] = beta1 * adam.m_center[j] + (1.0f - beta1) * grad.center[j];
        adam.v_center[j] = beta2 * adam.v_center[j] + (1.0f - beta2) * grad.center[j] * grad.center[j];
        param.center[j] -= lr_center_corrected * adam.m_center[j] / (sqrtf(adam.v_center[j]) + epsilon);
    }
    
    // Update scale
    for (int j = 0; j < 2; j++) {
        adam.m_scale[j] = beta1 * adam.m_scale[j] + (1.0f - beta1) * grad.scale[j];
        adam.v_scale[j] = beta2 * adam.v_scale[j] + (1.0f - beta2) * grad.scale[j] * grad.scale[j];
        param.scale[j] -= lr_scale_corrected * adam.m_scale[j] / (sqrtf(adam.v_scale[j]) + epsilon);
    }
    
    // Update rotation
    adam.m_rotation[0] = beta1 * adam.m_rotation[0] + (1.0f - beta1) * grad.rotation[0];
    adam.v_rotation[0] = beta2 * adam.v_rotation[0] + (1.0f - beta2) * grad.rotation[0] * grad.rotation[0];
    param.rotation[0] -= lr_rotation_corrected * adam.m_rotation[0] / (sqrtf(adam.v_rotation[0]) + epsilon);
    
    // Update color
    for (int j = 0; j < 3; j++) {
        adam.m_color[j] = beta1 * adam.m_color[j] + (1.0f - beta1) * grad.color[j];
        adam.v_color[j] = beta2 * adam.v_color[j] + (1.0f - beta2) * grad.color[j] * grad.color[j];
        param.color[j] -= lr_color_corrected * adam.m_color[j] / (sqrtf(adam.v_color[j]) + epsilon);
    }
    
    // Update opacity
    adam.m_opacity[0] = beta1 * adam.m_opacity[0] + (1.0f - beta1) * grad.opacity[0];
    adam.v_opacity[0] = beta2 * adam.v_opacity[0] + (1.0f - beta2) * grad.opacity[0] * grad.opacity[0];
    param.opacity[0] -= lr_opacity_corrected * adam.m_opacity[0] / (sqrtf(adam.v_opacity[0]) + epsilon);
}

void GaussianCollection::adam_step_gpu(float learning_rate, float beta1, float beta2, 
                                       float epsilon, int iteration) {
    float beta1_t = powf(beta1, iteration);
    float beta2_t = powf(beta2, iteration);
    
    // Calculate grid dimensions
    int block_size = 256;
    int grid_size = (num_gaussians + block_size - 1) / block_size;
    
    // Launch kernel
    adam_step_kernel<<<grid_size, block_size>>>(
        device_params.get(),
        device_grads.get(),
        device_adam.get(),
        num_gaussians,
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
}

void GaussianCollection::adam_step_gpu_individual(float lr_center, float lr_scale, float lr_rotation,
                                                  float lr_color, float lr_opacity,
                                                  float beta1, float beta2, float epsilon, int iteration) {
    float beta1_t = powf(beta1, iteration);
    float beta2_t = powf(beta2, iteration);
    
    // Calculate grid dimensions
    int block_size = 256;
    int grid_size = (num_gaussians + block_size - 1) / block_size;
    
    // Launch kernel
    adam_step_individual_kernel<<<grid_size, block_size>>>(
        device_params.get(),
        device_grads.get(),
        device_adam.get(),
        num_gaussians,
        lr_center,
        lr_scale,
        lr_rotation,
        lr_color,
        lr_opacity,
        beta1,
        beta2,
        epsilon,
        beta1_t,
        beta2_t
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Adam individual kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}