# Intro
We proposes a window-order-based convolution paradigm on GPU, called im2win, which not only reduces memory footprint but also offers continuous memory ac- cesses, resulting in improved performance. This project is an efficient convolution paradigm on GPU.
# Description
Our package has 4 parts: source code (src), benchmarks (benchmarks), log, and gnuplot for plotting figures (plot). Please refer to the Readme file in each part in the package.
# Prerequisites
CMake $>=$ 3.10 \
GCC $>=$ 7.5.0 \
PyTorch $==$ 1.10.0a0 \
CUDA $==$ 11.1 \
cuBLAS $==$ 11.2 \
cuDNN $==$ 8.0.1 \
gnuplot \
bash
# Basic Usage
Out of dir compilation:
<pre> $ cd im2win-CUDA
 $ mkdir build
 $ cd build
 $ cmake ../benchmarks
 $ make
 </pre>
 Or run the script:
 <pre>
 $ cd im2win-CUDA
 $ bash build.sh
 </pre>
 The compiled test benchmark can be run using the following command: 
 <pre>
 $ cd im2win-CUDA
 $ ./build/benchmarks
 </pre>
 After executing this command, different convolutional algorithms will be run on the test benchmark and performance will be recorded, with the results output to the log folder.
# Please cite
