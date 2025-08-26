#pragma once

#include <cuda_runtime.h>
#include "../../include/variable.cuh"
#include "../../include/operations/binary/add_logic.cuh"
#include "../../include/operations/binary/mul_logic.cuh"
#include "../../include/operations/binary/sub_logic.cuh"
#include "../../include/operations/unary/neg_logic.cuh"
#include "../../include/operations/unary/exp_logic.cuh"
#include "../../include/operations/unary/broadcast.cuh"
#include "../../include/operations/unary/l1_norm_logic.cuh"
#include "operations/covariance_generation.cuh"
#include "operations/mahalanobis_distance.cuh"
#include "../../include/operations/unary/sym_matrix2_inv_logic.cuh"
#include "gaussian_parameters.h"

using namespace xyz_autodiff;

// Tile size for CUDA blocks
constexpr int TILE_SIZE = 16;

// Structure to hold rendering output for one pixel
struct PixelOutput {
    float color[3];  // RGB
    float alpha;     // Alpha (accumulated opacity)
    float loss;      // Loss contribution from this pixel
};

// CUDA kernel for Gaussian splatting rendering and gradient computation
__global__ void gaussian_splatting_kernel(
    const GaussianParams* gaussians,     // Input: Gaussian parameters [NUM_GAUSSIANS]
    GaussianGrads* gradients,            // Output: Accumulated gradients [NUM_GAUSSIANS]
    const float* target_image,           // Input: Target image data [width * height * 3]
    PixelOutput* output_image,           // Output: Rendered image [width * height]
    float* total_loss,                   // Output: Atomic accumulator for total L1 loss
    int image_width,
    int image_height,
    int num_gaussians
);

// Host function to launch the kernel with proper tile configuration
void launch_gaussian_splatting(
    const GaussianParams* device_gaussians,
    GaussianGrads* device_gradients, 
    const float* device_target_image,
    PixelOutput* device_output_image,
    float* device_total_loss,
    int image_width,
    int image_height,
    int num_gaussians
);

// Fixed version with reduced memory pressure
void launch_gaussian_splatting_fixed(
    const GaussianParams* device_gaussians,
    GaussianGrads* device_gradients,
    const float* device_target_image, 
    PixelOutput* device_output_image,
    int image_width,
    int image_height,
    int num_gaussians
);

// Calculate total loss from pixel outputs
float calculate_total_loss(const PixelOutput* device_output, int image_width, int image_height);