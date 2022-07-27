cmake_minimum_required(VERSION 3.7)
project(demo)
set(SOURCES test.cpp
WeTensor.hpp)
add_executable(${PROJECT_NAME} ${SOURCES})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/ubuntu/code_cpp/TA/libtorch")
find_package(Torch REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx2 -mfma -march=native")#Rpass=.*
    set(CMAKE_EXE_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ../libtorch/include)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set(CMAKE_BUILD_TYPE "Debug")

target_include_directories(${PROJECT_NAME} PRIVATE /home/OpenBLAS/include/)
target_include_directories(${PROJECT_NAME} PRIVATE /home/ubuntu/code_cpp/TA/my_convolution)

target_link_libraries(${PROJECT_NAME} libopenblas.a)
target_link_libraries(${PROJECT_NAME} -lm)
target_link_libraries(${PROJECT_NAME} -lpthread)
