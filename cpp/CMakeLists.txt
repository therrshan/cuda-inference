cmake_minimum_required(VERSION 3.10)
project(CUDAInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    include
)

# CPU sources (CUDA disabled for now - will add later)
set(CPU_SOURCES
    src/tensor.cpp
    src/layers.cpp
    src/tokenizer.cpp
    src/model.cpp
    src/main.cpp
)

# Create executable
add_executable(inference ${CPU_SOURCES})

target_link_libraries(inference
    Eigen3::Eigen
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(inference OpenMP::OpenMP_CXX)
endif()