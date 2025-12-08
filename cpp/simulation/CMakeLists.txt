cmake_minimum_required(VERSION 3.18)

# Force CUDA 12.8 toolkit (driver supports up to 12.8, not 13.0)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")

project(CellSimulation LANGUAGES CXX CUDA)

# Build options
option(SAFE_MODE "Enable GPU memory tracking with 1GB limit (for debugging)" OFF)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA architecture (adjust for your GPU)
# Common values: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (A100), 86 (Ampere), 89 (Ada), 90 (Hopper/H100)
# RTX 4090 = compute capability 8.9 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cu
    src/kernels_shared.cu
    src/kernels_solver.cu
    src/kernels3d.cu
    src/integrator.cu
    src/io.cu
    src/io3d.cu
)

# Header files
set(HEADERS
    include/types.cuh
    include/types3d.cuh
    include/physics.cuh
    include/cell.cuh
    include/cell3d.cuh
    include/domain.cuh
    include/domain3d.cuh
    include/kernels.cuh
    include/kernels3d.cuh
    include/integrator.cuh
    include/integrator3d.cuh
    include/io.cuh
    include/simulation.cuh
    include/simulation3d.cuh
)

# Create executable
add_executable(cell_sim ${SOURCES} ${HEADERS})

# Link CUDA runtime and curand for GPU-side RNG
target_link_libraries(cell_sim CUDA::cudart CUDA::curand)

# Compiler flags
target_compile_options(cell_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -O3
    --use_fast_math
    >
)

# Safe mode: define SAFE_MODE macro for GPU memory tracking
if(SAFE_MODE)
    target_compile_definitions(cell_sim PRIVATE SAFE_MODE)
    message(STATUS "SAFE_MODE enabled: GPU memory tracking with 1GB limit")
endif()

# Debug configuration
target_compile_options(cell_sim PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
    -G
    -lineinfo
    >
)

# Output directory
set_target_properties(cell_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
