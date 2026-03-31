cmake_minimum_required(VERSION 3.18)
project(cell_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Common source files
set(COMMON_SOURCES
    src/cell.cpp
    src/domain.cpp
    src/integrator.cpp
    src/io.cpp
)

#==============================================================================
# Target 1: CPU-only version (OpenMP, no MPI)
#==============================================================================
add_executable(cell_sim_cpu
    src/main.cpp
    ${COMMON_SOURCES}
)

target_include_directories(cell_sim_cpu PRIVATE include)

# Compiler flags for CPU version
if(MSVC)
    target_compile_options(cell_sim_cpu PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(cell_sim_cpu PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# OpenMP for CPU parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cell_sim_cpu OpenMP::OpenMP_CXX)
    target_compile_definitions(cell_sim_cpu PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found - CPU version will use parallel loops")
else()
    message(STATUS "OpenMP not found - CPU version will be serial")
endif()

#==============================================================================
# Target 2: MPI version (with OpenMP for hybrid parallelization)
#==============================================================================
find_package(MPI)

if(MPI_CXX_FOUND)
    add_executable(cell_sim_mpi
        src/main_mpi.cpp
        src/mpi_spatial.cpp
        ${COMMON_SOURCES}
    )
    
    target_include_directories(cell_sim_mpi PRIVATE include)
    target_link_libraries(cell_sim_mpi MPI::MPI_CXX)
    target_compile_definitions(cell_sim_mpi PRIVATE USE_MPI)
    
    # Compiler flags for MPI version
    if(MSVC)
        target_compile_options(cell_sim_mpi PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(cell_sim_mpi PRIVATE -O3 -march=native -Wall -Wextra)
    endif()
    
    # OpenMP for hybrid parallelization
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cell_sim_mpi OpenMP::OpenMP_CXX)
        target_compile_definitions(cell_sim_mpi PRIVATE USE_OPENMP)
        message(STATUS "MPI+OpenMP: Hybrid parallelization enabled")
    endif()
    
    message(STATUS "MPI found - building cell_sim_mpi with spatial decomposition")
else()
    message(STATUS "MPI not found - only building cell_sim_cpu")
endif()
