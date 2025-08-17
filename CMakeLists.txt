cmake_minimum_required(VERSION 3.18)
project(xyz_autodiff VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

include(GNUInstallDirs)

add_library(xyz_autodiff
    src/tensor.cu
)

target_include_directories(xyz_autodiff
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(xyz_autodiff
    PUBLIC
        CUDA::cudart
)

set_target_properties(xyz_autodiff PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/xyz_autodiffConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xyz_autodiffConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xyz_autodiffConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xyz_autodiff
)

install(TARGETS xyz_autodiff
    EXPORT xyz_autodiffTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT xyz_autodiffTargets
    FILE xyz_autodiffTargets.cmake
    NAMESPACE xyz_autodiff::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xyz_autodiff
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/xyz_autodiffConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/xyz_autodiffConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xyz_autodiff
)