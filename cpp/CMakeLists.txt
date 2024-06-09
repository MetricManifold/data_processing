cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 14)

# specify the Eigen header files location
set(Eigen3_DIR "E:/etc/eigen3/share/eigen3/cmake")
set(FFTW3_DIR "E:/etc/fftw/lib/cmake/fftw3")

find_package(Eigen3 REQUIRED)
find_package(FFTW3 REQUIRED)

add_executable(MyProject main.cpp)
target_link_libraries(MyProject Eigen3::Eigen ${FFTW3_LIBRARIES})
target_link_directories(MyProject PUBLIC ${FFTW3_LIBRARY_DIRS})
target_include_directories(MyProject PRIVATE ${FFTW3_INCLUDE_DIRS})
