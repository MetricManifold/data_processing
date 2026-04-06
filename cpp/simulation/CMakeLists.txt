cmake_minimum_required(VERSION 3.18)

#==============================================================================
# Backend Selection
#==============================================================================
# Use -DBACKEND=<backend> to select:
#   CUDA   - GPU acceleration (default if CUDA found)
#   MPI    - CPU with MPI support for distributed computing
#   SERIAL - CPU-only, single process (OpenMP for thread parallelism)
#
# Examples:
#   cmake .. -DBACKEND=CUDA
#   cmake .. -DBACKEND=MPI
#   cmake .. -DBACKEND=SERIAL
#==============================================================================

set(BACKEND "AUTO" CACHE STRING "Backend: CUDA, MPI, SERIAL, or AUTO")
set_property(CACHE BACKEND PROPERTY STRINGS "AUTO" "CUDA" "MPI" "SERIAL")

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#==============================================================================
# Auto-detect backend if not specified
#==============================================================================
if(BACKEND STREQUAL "AUTO")
    # Try CUDA first
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(BACKEND "CUDA")
        message(STATUS "Auto-detected backend: CUDA")
    else()
        # Fall back to SERIAL (MPI requires explicit selection)
        set(BACKEND "SERIAL")
        message(STATUS "Auto-detected backend: SERIAL (CUDA not found)")
    endif()
endif()

#==============================================================================
# Project setup based on backend
#==============================================================================
if(BACKEND STREQUAL "CUDA")
    # CUDA backend
    if(WIN32)
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
        set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
        # Allow VS2026 (unsupported but may work)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    endif()
    project(CellSimulation LANGUAGES CXX CUDA)
    set(USE_CUDA ON)
    set(USE_MPI OFF)
else()
    # CPU backends (MPI or SERIAL)
    project(CellSimulation LANGUAGES CXX)
    set(USE_CUDA OFF)
    if(BACKEND STREQUAL "MPI")
        set(USE_MPI ON)
    else()
        set(USE_MPI OFF)
    endif()
endif()

#==============================================================================
# Build options
#==============================================================================
option(SAFE_MODE "Enable GPU memory tracking with 1GB limit (CUDA only)" OFF)
option(ENABLE_DIAGNOSTICS "Enable GPU-side diagnostics (CUDA only)" OFF)
option(ENABLE_STRESS_FIELDS "Enable stress tensor field output (CUDA only)" OFF)
option(USE_HALF_PRECISION_3D "Use FP16 for 3D phi fields (CUDA only)" OFF)
option(ENABLE_KERNEL_PROFILING "Enable per-phase timing (CUDA only)" OFF)
option(ENABLE_VISUALIZER "Enable real-time OpenGL visualizer (CUDA only)" OFF)

#==============================================================================
# C++ Standard
#==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==============================================================================
# CUDA Backend
#==============================================================================
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)
    endif()
    
    # Source files (CUDA)
    set(SOURCES
        src/main.cu
        src/kernels_shared.cu
        src/kernels_solver.cu
        src/kernels3d.cu
        src/integrator.cu
        src/io.cu
        src/io3d.cu
    )
    
    # Headers
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
    
    # Link CUDA libraries
    target_link_libraries(cell_sim CUDA::cudart CUDA::curand)
    
    # CUDA compile options
    target_compile_options(cell_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -O3
            --use_fast_math
            -allow-unsupported-compiler
        >
    )
    
    # Debug options
    target_compile_options(cell_sim PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
            -G
            -lineinfo
        >
    )
    
    # CUDA-specific definitions
    target_compile_definitions(cell_sim PRIVATE BACKEND_CUDA)
    
    if(SAFE_MODE)
        target_compile_definitions(cell_sim PRIVATE SAFE_MODE)
        message(STATUS "SAFE_MODE enabled")
    endif()
    
    if(ENABLE_DIAGNOSTICS)
        target_compile_definitions(cell_sim PRIVATE DIAGNOSTICS_ENABLED)
        message(STATUS "DIAGNOSTICS enabled")
    endif()
    
    if(ENABLE_STRESS_FIELDS)
        target_compile_definitions(cell_sim PRIVATE STRESS_FIELDS_ENABLED)
        message(STATUS "STRESS_FIELDS enabled")
    endif()
    
    if(USE_HALF_PRECISION_3D)
        target_compile_definitions(cell_sim PRIVATE USE_HALF_PRECISION_3D)
        message(STATUS "USE_HALF_PRECISION_3D enabled")
    endif()
    
    if(ENABLE_KERNEL_PROFILING)
        target_compile_definitions(cell_sim PRIVATE ENABLE_KERNEL_PROFILING=1)
        message(STATUS "ENABLE_KERNEL_PROFILING enabled")
    endif()
    
    if(ENABLE_VISUALIZER)
        # Fetch GLFW for windowing
        include(FetchContent)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4
        )
        FetchContent_MakeAvailable(glfw)

        # Find OpenGL
        find_package(OpenGL REQUIRED)

        target_sources(cell_sim PRIVATE src/visualizer.cu)
        target_link_libraries(cell_sim glfw OpenGL::GL)
        target_compile_definitions(cell_sim PRIVATE ENABLE_VISUALIZER=1)
        message(STATUS "VISUALIZER enabled (GLFW + OpenGL)")
    endif()
    
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

#==============================================================================
# MPI Backend
#==============================================================================
elseif(USE_MPI)
    find_package(MPI REQUIRED)
    
    # Source files (CPU/MPI)
    set(SOURCES
        src/cpu/main.cpp
        src/cpu/cell.cpp
        src/cpu/domain.cpp
        src/cpu/integrator.cpp
        src/cpu/io.cpp
    )
    
    # Create executable
    add_executable(cell_sim ${SOURCES})
    
    # Include directories
    target_include_directories(cell_sim PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu
    )
    
    # Link MPI
    target_link_libraries(cell_sim MPI::MPI_CXX)
    
    # Definitions
    target_compile_definitions(cell_sim PRIVATE BACKEND_MPI USE_MPI)
    
    # Compiler flags
    if(MSVC)
        target_compile_options(cell_sim PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(cell_sim PRIVATE -O3 -march=native -Wall -Wextra)
    endif()
    
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

#==============================================================================
# Serial Backend (CPU, no MPI)
#==============================================================================
else()
    # Source files (CPU/Serial)
    set(SOURCES
        src/cpu/main.cpp
        src/cpu/cell.cpp
        src/cpu/domain.cpp
        src/cpu/integrator.cpp
        src/cpu/io.cpp
    )
    
    # Create executable
    add_executable(cell_sim ${SOURCES})
    
    # Include directories
    target_include_directories(cell_sim PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu
    )
    
    # Definitions
    target_compile_definitions(cell_sim PRIVATE BACKEND_SERIAL)
    
    # Compiler flags
    if(MSVC)
        target_compile_options(cell_sim PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(cell_sim PRIVATE -O3 -march=native -Wall -Wextra)
    endif()
endif()

#==============================================================================
# Common settings for all backends
#==============================================================================

# Include directories (CUDA backend uses .cuh directly)
if(NOT USE_CUDA)
    target_include_directories(cell_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu)
endif()
target_include_directories(cell_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenMP for CPU backends (and hybrid for CUDA host code)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cell_sim OpenMP::OpenMP_CXX)
    target_compile_definitions(cell_sim PRIVATE USE_OPENMP)
    message(STATUS "OpenMP enabled")
endif()

# Output directory
set_target_properties(cell_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Summary
message(STATUS "========================================")
message(STATUS "Backend: ${BACKEND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
