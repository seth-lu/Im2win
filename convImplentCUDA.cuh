#include <cuda.h>
#include "cublas_v2.h"
#include <iostream>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>

void warmUp();

int testStream();

template <class dataType>
void directConvCUDAimplentNCHW(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
void directConvCUDAimplentNHWC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
void test_cuda(dataType input);

template <class dataType>
int im2winConvCUDAimplentNCHWBASE(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCBASE(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNCHWHPC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCHPC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCBLAS(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);