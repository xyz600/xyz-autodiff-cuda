# Mini Gaussian Splatting Example CMakeLists.txt

# Gaussian Splatting Training Application
add_executable(gaussian_splatting_training 
    gaussian_splatting_training.cu
    gaussian_parameters.cu
    image_utils.cpp
    gaussian_splatting_kernel.cu
    training_config.cpp
)
target_link_libraries(gaussian_splatting_training xyz_autodiff)
target_include_directories(gaussian_splatting_training PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)
set_target_properties(gaussian_splatting_training PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
# Add optimization flags
target_compile_options(gaussian_splatting_training PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=128>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=false>
)
