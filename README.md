# Im2win
code for paper Im2win: Memory-Efficient-Convolution-On-SIMD-Architectures.

We provide a C++ code implementation of the im2win convolution algorithm. All algorithm implementations use the at::Tensor data structure defined in libtorch.

  --WeTensor.hpp \ defined tensor data structure.
  --Convolution.hpp \ im2col-conv, direct-conv and im2win-conv.
  --im2winSIMD.hpp \ im2win convolution on SIMD instruction.
  --test_benchmark.cpp \ Twelve convolution layers of the DNN benchmarks.
