cmake_minimum_required(VERSION 3.18)
project(cell_sim_mpi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find MPI (optional for local development)
find_package(MPI)

# Create executable
add_executable(cell_sim_mpi
    src/main.cpp
    src/cell.cpp
    src/domain.cpp
    src/integrator.cpp
    src/io.cpp
)

target_include_directories(cell_sim_mpi PRIVATE include)

if(MPI_CXX_FOUND)
    target_link_libraries(cell_sim_mpi MPI::MPI_CXX)
    target_compile_definitions(cell_sim_mpi PRIVATE USE_MPI)
    message(STATUS "MPI found - building with MPI support")
else()
    message(STATUS "MPI not found - building serial version")
endif()

# Compiler flags - don't override CMAKE_BUILD_TYPE flags on MSVC
if(MSVC)
    target_compile_options(cell_sim_mpi PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(cell_sim_mpi PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Optional: OpenMP for hybrid parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cell_sim_mpi OpenMP::OpenMP_CXX)
    target_compile_definitions(cell_sim_mpi PRIVATE USE_OPENMP)
endif()
