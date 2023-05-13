# Intro
This project is an efficient convolution paradigm on GPU.
# Description
Our package has 4 parts: source code (src), unit test (test), log, and gnuplot for plotting figures (plot). Please refer to the Readme file in each part in the package.
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
 $ cmake ..
 $ make
 </pre>
 Or run the script:
 <pre>
 $ cd im2win-CUDA
 $ bash build.sh
 </pre>
# Please cite
