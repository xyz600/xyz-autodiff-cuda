#pragma once

#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

namespace xyz_autodiff {}
using namespace xyz_autodiff;

// Single Gaussian parameter structure
struct GaussianParams {
    float center[2];        // 2D center position
    float scale[2];         // 2D scale (standard deviation)
    float rotation[1];      // rotation angle
    float color[3];         // RGB color
    float opacity[1];       // opacity
};

// Adam optimizer state for a single Gaussian
struct AdamState {
    float m_center[2];      // momentum for center
    float v_center[2];      // velocity for center
    float m_scale[2];       // momentum for scale
    float v_scale[2];       // velocity for scale
    float m_rotation[1];    // momentum for rotation
    float v_rotation[1];    // velocity for rotation
    float m_color[3];       // momentum for color
    float v_color[3];       // velocity for color
    float m_opacity[1];     // momentum for opacity
    float v_opacity[1];     // velocity for opacity
};

// Gradient structure matching GaussianParams
struct GaussianGrads {
    float center[2];
    float scale[2];
    float rotation[1];
    float color[3];
    float opacity[1];
};

// Forward declaration
struct TrainingConfig;

// Container for all Gaussian parameters and optimization state
class GaussianCollection {
public:
    int num_gaussians;
    
    // Host data
    std::vector<GaussianParams> host_params;
    std::vector<GaussianGrads> host_grads;
    std::vector<AdamState> host_adam;
    
    // Device data (using CUDA unique pointers)
    cuda_unique_ptr<GaussianParams[]> device_params;
    cuda_unique_ptr<GaussianGrads[]> device_grads;
    cuda_unique_ptr<AdamState[]> device_adam;
    
    GaussianCollection(const TrainingConfig& config);
    
    // Legacy constructor for backward compatibility  
    GaussianCollection() : num_gaussians(1000) {}
    
    // Initialize Gaussians with random parameters
    void initialize_random(int image_width, int image_height, std::mt19937& rng);
    
    // Copy data between host and device
    void upload_to_device();
    void download_from_device();
    
    // Clear gradients
    void zero_gradients();
    
    // Apply Adam optimization step
    void adam_step(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, 
                   float epsilon = 1e-8f, int iteration = 1);
    
    // Clear gradients on device
    void zero_gradients_gpu();
    
    // Apply GPU Adam optimization step (replaces host-side adam_step)
    void adam_step_gpu(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, 
                       float epsilon = 1e-8f, int iteration = 1);
    
    // Apply GPU Adam optimization step with individual learning rates
    void adam_step_gpu_individual(float lr_center, float lr_scale, float lr_rotation,
                                  float lr_color, float lr_opacity,
                                  float beta1 = 0.9f, float beta2 = 0.999f, 
                                  float epsilon = 1e-8f, int iteration = 1);
};