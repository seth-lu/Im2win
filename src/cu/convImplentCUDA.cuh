#include <cuda.h>
#include "cublas_v2.h"
#include <cudnn.h>
#include <iostream>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>

#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <class dataType>
struct SharedMemory{
    __device__ dataType *getPointer(){
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <>
struct SharedMemory<float>{
    __device__ float *getPointer(){
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory<double>{
    __device__ double *getPointer(){
        extern __shared__ double d_float[];
        return d_float;
    }
};

void warmUp();

int testStream();

template <class dataType>
void directConvCUDAimplentNCHW(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
void directConvCUDAimplentNHWC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
void implicitConvCUDAimplentNHWC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
void test_cuda(dataType input);

template <class dataType>
int im2winConvCUDAimplentNCHWBASE(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCBASE(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNCHWHPC_vectorized(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNCHWHPC_prefetch(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNCHWHPC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCHPC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCBLAS(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int im2winConvCUDAimplentNHWCBLASHPC(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);

template <class dataType>
int cuDNNConvNCHW(dataType *inptr, dataType *filptr, dataType *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c);