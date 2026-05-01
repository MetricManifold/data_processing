cmake_minimum_required(VERSION 3.18)

# ============================================================================
# cell_sim - phase-field cell simulation (neighbor-list method, post-cutover)
# ----------------------------------------------------------------------------
# As of 2026-04-23 this tree IS sim_v2 (cutover from baseline). The original
# baseline is preserved under git tag `baseline-v1-final`.
# ============================================================================

# Windows: point at CUDA 13.0 explicitly (must precede project()).
if(WIN32 AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
    # VS2026 is unsupported by nvcc but works in practice.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

project(cell_sim LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Covers Ampere (A100 sm_80, RTX 30xx sm_86), Ada (RTX 40xx sm_89), Hopper
# (H100 sm_90). Single list works on Windows laptop + Fir + Nibi + Narval +
# Rorqual. Override with `-DCMAKE_CUDA_ARCHITECTURES=<x>` for one-off builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optional CUDA + OpenGL live viewer. OFF by default. Enable locally with
#   cmake -DENABLE_VISUALIZER=ON ...
# Requires GLFW3 + OpenGL on the build host. SHOULD NOT BE ENABLED ON
# CLUSTER BUILDS (headless nodes, no display).
option(ENABLE_VISUALIZER "Build the live CUDA-OpenGL viewer (--live-view)" OFF)

set(CELL_SIM_SOURCES
    src/main.cu
    src/kernels.cu
    src/sim.cu
)
if(ENABLE_VISUALIZER)
    list(APPEND CELL_SIM_SOURCES src/visualizer.cu)
endif()

add_executable(cell_sim ${CELL_SIM_SOURCES})

target_include_directories(cell_sim PRIVATE include)
target_compile_options(cell_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr>
)

if(ENABLE_VISUALIZER)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    target_compile_definitions(cell_sim PRIVATE ENABLE_VISUALIZER)
    target_link_libraries(cell_sim PRIVATE glfw OpenGL::GL)
    message(STATUS "cell_sim: live viewer ENABLED (--live-view available)")
endif()
set_target_properties(cell_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_SEPARABLE_COMPILATION OFF
)
