cmake_minimum_required(VERSION 3.10)
project(benchmarks)

file( GLOB  cu  ../src *.cu *.cuh)
file( GLOB  hdr ../src *.hpp *.h)
set(SOURCES benchmarks.cpp)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/ubuntu/code_cpp/TA/libtorch-CUDA")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES} ${hdr})

set(CMAKE_CUDA_FLAGS "-O3")
set(CMAKE_NVCC_FLAGS "-O3")

enable_language(CUDA)
CUDA_ADD_LIBRARY(codeCUDA ${cu})
target_link_libraries(${PROJECT_NAME} codeCUDA)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set(CMAKE_BUILD_TYPE "Debug")

target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cudnn_LIBRARY})

