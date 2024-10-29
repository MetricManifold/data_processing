cmake_minimum_required(VERSION 3.10)
project(MatrixAdditionCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Specify the CUDA source files
set(CUDA_SOURCES main.cu)

add_library(MatrixAdditionO OBJECT cuda2.cu header2.cpp)

add_library(MatrixAdditionOLib STATIC)
target_link_libraries(MatrixAdditionOLib PUBLIC MatrixAdditionO)

add_library(MatrixAdditionO2 OBJECT cuda.cu header.cpp)
target_link_libraries(MatrixAdditionO2 PUBLIC MatrixAdditionOLib)

add_library(MatrixAdditionO2Lib STATIC)
target_link_libraries(MatrixAdditionO2Lib PUBLIC MatrixAdditionO2)

# Set the CUDA architecture (optional, adjust as needed)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")

# Add the executable
add_executable(MatrixAdditionImpl main.cu)
target_link_libraries(MatrixAdditionImpl PUBLIC MatrixAdditionO2Lib)