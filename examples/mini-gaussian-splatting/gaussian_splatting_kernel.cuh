#pragma once

#include <cuda_runtime.h>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/operations/binary/sub_logic.cuh>
#include <xyz_autodiff/operations/unary/neg_logic.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/operations/unary/broadcast.cuh>
#include <xyz_autodiff/operations/unary/l1_norm_logic.cuh>
#include "operations/covariance_generation.cuh"
#include "operations/mahalanobis_distance.cuh"
#include <xyz_autodiff/operations/unary/sym_matrix2_inv_logic.cuh>
#include "gaussian_parameters.h"
#include <xyz_autodiff/const_array.cuh>

using namespace xyz_autodiff;

// Tile size for CUDA blocks
constexpr int TILE_SIZE = 16;

using PixelOutput = ConstArray<float, 3>;

// CUDA kernel for Gaussian splatting rendering and gradient computation
__global__ void gaussian_splatting_kernel(
    const GaussianParams* gaussians,     // Input: Gaussian parameters [NUM_GAUSSIANS]
    GaussianGrads* gradients,            // Output: Accumulated gradients [NUM_GAUSSIANS]
    const PixelOutput* target_image,     // Input: Target image data [width * height]
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
    const PixelOutput* device_target_image,
    PixelOutput* device_output_image,
    float* device_total_loss,
    int image_width,
    int image_height,
    int num_gaussians
);
