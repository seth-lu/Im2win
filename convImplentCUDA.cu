// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
#include "convImplentCUDA.cuh"

static inline cublasStatus_t cublasXgemv(cublasHandle_t handle, cublasOperation_t trans,
                                         int m, int n, 
                                         const float *alpha,
                                         const float *a, int lda,
                                         const float *x, int incx,
                                         const float *beta,
                                         float *y, int incy){
    return  cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, a, lda,
                            x, incx, beta, y, incy);
}

static inline cublasStatus_t cublasXgemv(cublasHandle_t handle, cublasOperation_t trans,
                                         int m, int n, 
                                         const double *alpha,
                                         const double *a, int lda,
                                         const double *x, int incx,
                                         const double *beta,
                                         double *y, int incy){
    return  cublasDgemv(handle, CUBLAS_OP_T, m, n, alpha, a, lda,
                            x, incx, beta, y, incy);
}

__global__ void warmUpKernel(){}

void warmUp(){
    warmUpKernel<<<1, 1>>>();
    return;
}

__global__ void directKernelConvNCHW(float *d_inptr, float *d_filptr, float *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    Idx = Idx_z * output_height * output_width + Idx_x * output_width+ Idx_y;
    float tmp = 0.0;
    for(size_t c = 0; c < filter_channel; ++c){
        for(size_t x = 0; x < filter_height; ++x){
            for(size_t y = 0; y < filter_width; ++y){
                tmp += d_inptr[(Idx_x * stride_height + x) * input_width + Idx_y * stride_width + y + c * input_height * input_width]
                     * d_filptr[x * filter_width + y + c * filter_height * filter_width + Idx_z * filter_channel * filter_height * filter_width];
            }
        }
    }
    d_outptr[Idx] = tmp;
    // printf("id : %d\n", Idx);
}

template<>
void directConvCUDAimplentNCHW(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1, dims_c[1]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));

    for(size_t i = 0; i < dims_a[0]; ++i)
        directKernelConvNCHW<<<gridSize, blockSize>>>(inptr+i*dims_a[1]*dims_a[2]*dims_a[3], filptr, outptr+i*dims_c[1]*dims_c[2]*dims_c[3], d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

__global__ void directKernelConvNCHW(double *d_inptr, double *d_filptr, double *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    Idx = Idx_z * output_height * output_width + Idx_x * output_width+ Idx_y;
    double tmp = 0.0;
    for(size_t c = 0; c < filter_channel; ++c){
        for(size_t x = 0; x < filter_height; ++x){
            for(size_t y = 0; y < filter_width; ++y){
                tmp += d_inptr[(Idx_x * stride_height + x) * input_width + Idx_y * stride_width + y + c * input_height * input_width]
                     * d_filptr[x * filter_width + y + c * filter_height * filter_width + Idx_z * filter_channel * filter_height * filter_width];
            }
        }
    }
    d_outptr[Idx] = tmp;
    // printf("id : %d\n", Idx);
}

template<>
void directConvCUDAimplentNCHW(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    //double *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(double)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(double)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(double)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(double)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(double)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1);
    //run kernelConv
    directKernelConvNCHW<<<gridSize, blockSize>>>(inptr, filptr, outptr, dims_a, dims_b, dims_c);
    //result to host
    cudaMemcpy(outptr, outptr, sizeof(double)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

__global__ void directKernelConvNHWC(float *d_inptr, float *d_filptr, float *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    Idx = Idx_x * output_width * output_channel + Idx_y * output_channel + Idx_z;
    float tmp = 0.0;
    for(size_t x = 0; x < filter_height; ++x){
        for(size_t y = 0; y < filter_width; ++y){
            for(size_t c = 0; c < filter_channel; ++c){
                // printf("Idx : %d, D_V : %d\n", Idx, d_inptr[Idx_x * stride_height * input_width * input_channel + Idx_y * stride_width * input_channel + x * input_width * input_channel + y * input_channel + c]);
                tmp += d_inptr[Idx_x * stride_height * input_width * input_channel + Idx_y * stride_width * input_channel + x * input_width * input_channel + y * input_channel + c]
                     * d_filptr[x * filter_width * filter_channel + y * filter_channel + c + Idx_z * filter_channel * filter_height * filter_width];
            }
        }
    }
    d_outptr[Idx] = tmp;
    //printf("id : %d\n", Idx);
}

template<>
void directConvCUDAimplentNHWC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, 
                        (dims_c[3] - 1)/blockSize.y + 1, 
                         dims_c[1]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));
    // 
    // int dev = 0;
    // cudaSetDevice(dev);
    // size_t n_stream = 16;
    // cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    // for(size_t i = 0; i < n_stream; ++i){
    //     cudaStreamCreate(&stream[i]);
    // }

    for(size_t i = 0; i < dims_a[0]; ++i){
        directKernelConvNHWC<<<gridSize, blockSize>>>(inptr+i*dims_a[1]*dims_a[2]*dims_a[3], 
                                                      filptr,
                                                      outptr+i*dims_c[1]*dims_c[2]*dims_c[3], 
                                                      d_dims_a, d_dims_b, d_dims_c);
    }

    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

__global__ void directKernelConvNHWC(double *d_inptr, double *d_filptr, double *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    Idx = Idx_x * output_width * output_channel + Idx_y * output_channel + Idx_z;
    double tmp = 0.0;
    for(size_t x = 0; x < filter_height; ++x){
        for(size_t y = 0; y < filter_width; ++y){
            for(size_t c = 0; c < filter_channel; ++c){
                tmp += d_inptr[Idx_x * stride_height * input_width * input_channel + Idx_y * stride_width * input_channel + x * input_width * input_channel + y * input_channel + c]
                     * d_filptr[x * filter_width * filter_channel + y * filter_channel + c + Idx_z * filter_channel * filter_height * filter_width];
            }
        }
    }
    d_outptr[Idx] = tmp;
    // printf("id : %d\n", Idx);
}

template<>
void directConvCUDAimplentNHWC(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    //double *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(double)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(double)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(double)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(double)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(double)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1);
    //run kernelConv
    directKernelConvNHWC<<<gridSize, blockSize>>>(inptr, filptr, outptr, dims_a, dims_b, dims_c);
    //result to host
    cudaMemcpy(outptr, outptr, sizeof(double)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
void test_cuda<int>(int input){
    std::cout<<"run complete"<<std::endl;
    return;
}

__global__ void im2winKernelConvNCHWBASE(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    size_t indexY = Idx_z * output_height * output_width
                    + Idx_x * output_width + Idx_y;
    float tmp = 0.0;
    
    for(size_t c = 0; c < input_channel; ++c){
        for(size_t w = 0; w < filter_width; ++w){
            for(size_t h = 0; h < filter_height; ++h){
                size_t indexD = c *output_height * filter_height * input_width
                                + w * filter_height + h
                                + Idx_x * filter_height * input_width
                                + Idx_y * stride_width * filter_height;
                size_t indexW = c * filter_height * filter_width
                                + w * filter_height + h
                                + Idx_z * filter_channel * filter_height * filter_width;
                tmp += inptr[indexD] * filptr[indexW];
            }
        }
    }
    outptr[indexY] = tmp;
}

template<>
int im2winConvCUDAimplentNCHWBASE(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1, dims_c[1]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));

    for(size_t i = 0; i < dims_a[0]; ++i)
        im2winKernelConvNCHWBASE<<<gridSize, blockSize>>>(inptr+i*dims_a[1]*dims_c[2]*dims_b[2]*dims_a[3], filptr, outptr+i*dims_c[1]*dims_c[2]*dims_c[3], d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}



template<>
int im2winConvCUDAimplentNCHWBASE(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

__global__ void im2winKernelConvNHWCBASE(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, 
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    Idx_z = blockIdx.z;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    if(Idx_x >= output_height || Idx_y >= output_width) return;
    
    size_t indexY = Idx_z
                    + Idx_x * output_width * output_channel 
                    + Idx_y * output_channel;
    float tmp = 0.0;
    
    for(size_t w = 0; w < filter_width; ++w){
        for(size_t h = 0; h < filter_height; ++h){
            for(size_t c = 0; c < input_channel; ++c){
                size_t indexD = c
                                + w * filter_height * input_channel 
                                + h * input_channel
                                + Idx_x * filter_height * input_width * input_channel
                                + Idx_y * stride_width * filter_height * input_channel;
                size_t indexW = c 
                                + w * filter_height * filter_channel 
                                + h * filter_channel
                                + Idx_z * filter_channel * filter_height * filter_width;
                tmp += inptr[indexD] * filptr[indexW];
            }
        }
    }
    outptr[indexY] = tmp;
}

template<>
int im2winConvCUDAimplentNHWCBASE(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    const dim3 blockSize(32, 32);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1, dims_c[1]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));

    for(size_t i = 0; i < dims_a[0]; ++i)
        im2winKernelConvNHWCBASE<<<gridSize, blockSize>>>(inptr+i*dims_a[1]*dims_c[2]*dims_b[2]*dims_a[3], filptr, outptr+i*dims_c[1]*dims_c[2]*dims_c[3], d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
    return 0;
}

template<>
int im2winConvCUDAimplentNHWCBASE(double *inptr, double *filptr, double *outptr, 
                                  size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

__global__ void im2winKernelConvNCHWHPC(float *inptr, float *filptr, float *outptr,
                                        size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c, 
                                        size_t BLOCK, size_t CHUNK){
    size_t Idx, TX, TY, TZ, BX, BY, BZ, Idx_ON, Idx_OC, Idx_OH,
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    TX = threadIdx.x;
    TY = threadIdx.y;
    // TZ = threadIdx.z;
    BX = blockIdx.x;
    BY = blockIdx.y;
    BZ = blockIdx.z;

    //Idx_x = BX; //dims_c[0]
    // Idx_y = blockDim.y * BY+ TY;
    // Idx_z = blockDim.z * BZ + TZ;
    //Idx_y = BY; //dims_c[1]
    //Idx_z = BZ;
    Idx_ON = blockDim.x * BX + TX;
    Idx_OC = blockDim.y * BY + TY;
    Idx_OH = BZ;

    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);

    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    //if(Idx_x >= output_batch || Idx_y >= output_channel || Idx_z >= output_height) return;
    // if(BZ == (blockDim.z - 1)){
    //     size_t bound_x, bound_y;
    //     bound_x = output_height - BZ * BLOCK;
    //     bound_y = output_width - BZ * BLOCK;
    //     if(TX >= bound_x || TY >= bound_y) return;
    // }
    if(Idx_ON >= output_batch || Idx_OC >= output_channel) return;
    
    //printf("Idx_OW : %d\n", Idx_OW);
    // Idx = Idx_x * blockDim.x
    //      +Idx_y * blockDim.x
    //      +Idx_z * blockDim.x
    //      +TX;


        //+ BZ * BLOCK * BLOCK;
    //printf("indexY : %d\n", indexY);
    //if(indexY >= output_csize) return;
    size_t offset_Y = Idx_ON * output_channel * output_height * output_width
                + Idx_OC * output_height * output_width
                + Idx_OH * output_width;
    size_t offset_D = Idx_ON * input_channel * output_height * filter_height * input_width
                + Idx_OH * filter_height * input_width;
    size_t offset_W = Idx_OC * filter_channel * filter_height * filter_width;

    #pragma unroll
    for(size_t j = 0; j < output_width; ++j){
        size_t indexY = offset_Y + j;
        float tmp = 0;
        #pragma unroll
        for(size_t c = 0; c < input_channel; ++c){
            __shared__ float filter_share[1024];
            __shared__ float input_share[1024];
            if(TX < filter_height && TY < filter_width){
               filter_share[TY * filter_height + TX] = filptr[TY * filter_height + TX + offset_W + c * filter_height * filter_width];
               //printf("ID :%d, filptr :%f\n", TY * filter_height + TX, filptr[TY * filter_height + TX]);
                                                              //+ offset_W + c * filter_height * filter_width];
            }
            __syncthreads();
            #pragma unroll
            for(size_t w = 0; w < filter_width; ++w){  
                #pragma unroll
                for(size_t h = 0; h < filter_height; ++h){
                    size_t indexD = offset_D
                                    + c * output_height * filter_height * input_width
                                    + w * filter_height + h
                                    + j * stride_width * filter_height;
                                    
                    size_t indexW = offset_W
                                    + c * filter_height * filter_width                      
                                    + w * filter_height + h;

                    tmp += inptr[indexD] * filter_share[w * filter_height + h];
                    //tmp += inptr[indexD] * filptr[indexW];
                    //printf("shareF :%f\n", filter_share[w * filter_height + h]);
                }
            }
        }
        __syncthreads();
        outptr[indexY] = tmp;
    }
}

template<>
int im2winConvCUDAimplentNCHWHPC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    size_t BLOCK = 16;
    const dim3 blockSize(BLOCK, BLOCK);
    const dim3 gridSize((dims_c[0] - 1)/BLOCK + 1, (dims_c[1] - 1)/BLOCK + 1, dims_c[2]);
    size_t CHUNK = (dims_c[2] - 1)/BLOCK + 1;
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));


    im2winKernelConvNCHWHPC<<<gridSize, blockSize>>>(inptr, filptr, outptr, 
                                                     d_dims_a, d_dims_b, d_dims_c, BLOCK, CHUNK);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
int im2winConvCUDAimplentNCHWHPC(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

template <size_t BLOCK>
__global__ void im2winKernelConvNHWCHPC(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t Idx_x, Idx_y, Idx_z, Idx, TX, TY, BX, BY,
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    input_batch = d_dims_a[0];
    input_channel = d_dims_a[1];
    input_height = d_dims_a[2];
    input_width = d_dims_a[3];

    filter_batch = d_dims_b[0];
    filter_channel = d_dims_b[1];
    filter_height = d_dims_b[2];
    filter_width = d_dims_b[3];

    output_batch = d_dims_c[0];
    output_channel = d_dims_c[1];
    output_height = d_dims_c[2];
    output_width = d_dims_c[3];

    stride_height = (input_height - filter_height) / (output_height - 1);
    stride_width = (input_width - filter_width) / (output_width - 1);

    TX = threadIdx.x;
    TY = threadIdx.y;
    BX = blockIdx.x;
    BY = blockIdx.y;

    Idx_x = blockDim.x * blockIdx.x + threadIdx.x; //dims_c[2]
    Idx_y = blockDim.y * blockIdx.y + threadIdx.y; //dims_c[3]
    Idx_z = blockIdx.z; //dims_c[0]

    if(Idx_x >= output_height || Idx_y >= output_width) return;

    outptr += Idx_z * output_channel * output_height * output_width;
    inptr += Idx_z * input_channel * output_height * filter_height * input_width;

    float *begin_inptr = inptr + BY * BLOCK * input_channel;
    float *begin_filter = filptr;
    float *end_inptr = begin_filter + input_channel;

    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
    //
    
    for(float *in_ptr = begin_inptr, *fil_ptr = begin_filter; in_ptr < end_inptr;
        in_ptr += BLOCK, fil_ptr += BLOCK){
        __shared__ float inshare[BLOCK][BLOCK];
        __shared__ float filshare[BLOCK];

        inshare[TY][TX] = in_ptr[TY * input_channel + TX];
        filshare[TX] = filshare[TX];
    }

    for(size_t n = 0; n < filter_batch; ++n){
        size_t indexY = n
                        + Idx_x * output_width * output_channel 
                        + Idx_y * output_channel;
        float tmp = 0.0;
        for(size_t w = 0; w < filter_width; ++w){
            for(size_t h = 0; h < filter_height; ++h){
                for(size_t c = 0; c < input_channel; ++c){
                    size_t indexD = c
                                    + w * filter_height * input_channel 
                                    + h * input_channel
                                    + Idx_x * filter_height * input_width * input_channel
                                    + Idx_y * stride_width * filter_height * input_channel;
                    size_t indexW = c 
                                    + w * filter_height * filter_channel 
                                    + h * filter_channel
                                    + n * filter_channel * filter_height * filter_width;
                    tmp += inptr[indexD] * filptr[indexW];
                }
            }
        }
        outptr[indexY] = tmp;
    }
}

template<>
int im2winConvCUDAimplentNHWCHPC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
    //alloca memory
    // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
    // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
    // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
    cudaMalloc(&d_dims_a, sizeof(size_t)*4);
    cudaMalloc(&d_dims_b, sizeof(size_t)*4);
    cudaMalloc(&d_dims_c, sizeof(size_t)*4);
    //init input and filter
    // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
    //set blockSize and gridSize
    constexpr size_t BLOCK = 32;
    const dim3 blockSize(BLOCK, BLOCK);
    const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1, dims_c[0]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));
    im2winKernelConvNHWCHPC<BLOCK><<<gridSize, blockSize>>>(inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
    return 0;
}

template<>
int im2winConvCUDAimplentNHWCHPC(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

template<>
int im2winConvCUDAimplentNHWCBLAS(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    const size_t input_batch = dims_a[0];
    const size_t input_channel = dims_a[1];
    const size_t input_height = dims_a[2];
    const size_t input_width = dims_a[3];

    const size_t filter_batch = dims_b[0];
    const size_t filter_channel = dims_b[1];
    const size_t filter_height = dims_b[2];
    const size_t filter_width = dims_b[3];

    const size_t output_batch = dims_c[0];
    const size_t output_channel = dims_c[1];
    const size_t output_height = dims_c[2];
    const size_t output_width = dims_c[3];    

    const size_t stride_height = (dims_a[2] - dims_b[2]) / (dims_c[2] - 1);
    const size_t stride_width = (dims_a[3] - dims_b[3]) / (dims_c[3] - 1);

    const size_t output_csize = output_batch * output_channel * output_height * output_width;
    const size_t output_area = output_height * output_width;
    const size_t filter_area = filter_height * filter_width;
    const size_t filter_volume = filter_area * filter_channel;
    const size_t gap_width = stride_width * filter_height;
    const size_t window_row = filter_height * input_width;
    const size_t window_area = output_height * window_row;
    const size_t window_volume = input_channel * window_area;
    const size_t output_volume = filter_batch * output_area;

    int m, n, lda, incx, incy;
    float alpha, beta, *a, *x, *y;
    m = filter_height * filter_width * input_channel;
    n = filter_batch;
    lda = m;
    incx = 1;
    incy = 1;
    alpha = 1;
    beta = 1;

    // #pragma omp parallel for schedule(omp_flag)
    // parallel by stream

    size_t n_stream = (input_batch - 1) / 8 + 1;
    cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    cublasHandle_t *handle = (cublasHandle_t *)malloc(n_stream * sizeof(cublasHandle_t));
    for(size_t i = 0; i < n_stream; ++i){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
        cublasSetStream(handle[i], stream[i]);
    }
    size_t id_stream = 0;

    nvtxRangePush(__func__);
    
    for(int b = 0; b < input_batch; ++b){
        int bY = b * output_height * output_width * filter_batch;
        int bD = b * output_height * filter_height * input_width * input_channel;
        for(int i = 0; i < output_height; ++i){
            int ibY = bY + i * output_width * filter_batch;
            int ibD = bD + i * filter_height * input_width * input_channel;
            #pragma unroll
            for(int j = 0; j < output_width; ++j){
                int jibY = ibY + j * filter_batch;
                int jcibD = ibD + j * stride_width * filter_height * input_channel;

                a = filptr;
                x = inptr + jcibD;
                y = outptr + jibY;

                nvtxRangePush("cublasSgemv call");
                
                cublasXgemv(handle[id_stream], CUBLAS_OP_T, m, n, &alpha, a, lda,
                            x, incx, &beta, y, incy);

                
                nvtxRangePop();
            }
            nvtxRangePop();
        }
        if((b+1)%8==0)id_stream++;
        nvtxRangePop();
    }
    nvtxRangePop();      
    return 0;
}

template<>
int Deprecate_im2winConvCUDAimplentNHWC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    //std::cout<<"im2winNHWC run"<<std::endl;
    if(status != CUBLAS_STATUS_SUCCESS){
        if(status == CUBLAS_STATUS_NOT_INITIALIZED){
            std::cout<<"CUBLAS status error!"<<std::endl;
        }
        getchar();
        return EXIT_FAILURE;
    }

    const size_t input_batch = dims_a[0];
    const size_t input_channel = dims_a[1];
    const size_t input_height = dims_a[2];
    const size_t input_width = dims_a[3];

    const size_t filter_batch = dims_b[0];
    const size_t filter_channel = dims_b[1];
    const size_t filter_height = dims_b[2];
    const size_t filter_width = dims_b[3];

    const size_t output_batch = dims_c[0];
    const size_t output_channel = dims_c[1];
    const size_t output_height = dims_c[2];
    const size_t output_width = dims_c[3];    

    const size_t stride_height = (dims_a[2] - dims_b[2]) / (dims_c[2] - 1);
    const size_t stride_width = (dims_a[3] - dims_b[3]) / (dims_c[3] - 1);

    const size_t output_csize = output_batch * output_channel * output_height * output_width;
    const size_t output_area = output_height * output_width;
    const size_t filter_area = filter_height * filter_width;
    const size_t filter_volume = filter_area * filter_channel;
    const size_t gap_width = stride_width * filter_height;
    const size_t window_row = filter_height * input_width;
    const size_t window_area = output_height * window_row;
    const size_t window_volume = input_channel * window_area;
    const size_t output_volume = filter_batch * output_area;

    int m, n, lda, incx, incy, batchCount;
    float alpha, beta, **a, **x, **y;
    batchCount = input_batch * output_height * output_width;
    m = filter_height * filter_width * input_channel;
    n = filter_batch;
    lda = m;
    incx = 1;
    incy = 1;
    alpha = 1;
    beta = 1;
    a = (float **)malloc(sizeof(float*)*batchCount);
    x = (float **)malloc(sizeof(float*)*batchCount);
    y = (float **)malloc(sizeof(float*)*batchCount);
    // #pragma omp parallel for schedule(omp_flag)
    // parallel by stream
    // size_t n_stream = input_batch * output_height * output_width;
    // cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    // for(size_t i = 0; i < n_stream; ++i)
    //     cudaStreamCreate(&stream[i]);
    // size_t id_stream = 0;

    //Create pointers of Aarray, Xarray and Yarray;
    int count = 0;
    for(int b = 0; b < input_batch; ++b){
        int bY = b * output_height * output_width * filter_batch;
        int bD = b * output_height * filter_height * input_width * input_channel;
        for(int i = 0; i < output_height; ++i){
            int ibY = bY + i * output_width * filter_batch;
            int ibD = bD + i * filter_height * input_width * input_channel;
            for(int j = 0; j < output_width; ++j){
                int jibY = ibY + j * filter_batch;
                int jcibD = ibD + j * stride_width * filter_height * input_channel;

                a[count] = filptr;
                x[count] = inptr + jcibD;
                y[count] = outptr + jibY;
                count++;
            }
        }
    }
    cublasSgemvBatched(handle, CUBLAS_OP_T, m, n, &alpha, a, lda, x, incx,
                       &beta, y, incy, batchCount);
    return 0;
}

// __global__ void deprecated_im2winKernelConvNHWC(float *inptr, float *filptr, float *outptr, size_t *d_size_outputHW, size_t *d_szie_A, size_t *d_offset_D, size_t *d_offset_Y){
//     size_t Idx_x, Idx_y, Idx_z, Idx;
    
//     Idx_x = blockDim.x * blockIdx.x + threadIdx.x;
//     Idx_y = blockDim.y * blockIdx.y + threadIdx.y;
//     Idx_z = blockIdx.z;

//     //printf("Idx_x : %d\n", Idx_x);
//     // printf("Idx_y : %d\n", Idx_y);
//     if(Idx_x >= d_size_outputHW[0] || Idx_y >= d_size_outputHW[1]) return;
    
//     cublasHandle_t handle;
//     cublasStatus_t status = cublasCreate(&handle);

//     int m, n, lda, incx, incy;
//     float alpha, beta, *a, *x, *y;
//     m = d_szie_A[0];
//     n = d_szie_A[1];
//     lda = m;
//     incx = 1;
//     incy = 1;
//     alpha = 1;
//     beta = 1;
//     // #pragma omp parallel for schedule(omp_flag)

//     int jibY =  Idx_x * d_offset_Y[0] + Idx_y * d_offset_Y[1] + Idx_z * d_offset_Y[2];
//     int jcibD = Idx_x * d_offset_D[0] + Idx_y * d_offset_D[1] + Idx_z * d_offset_D[2];

//     a = filptr;
//     x = inptr + jcibD;
//     y = outptr + jibY;

//     cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, a, lda,
//                             x, incx, &beta, y, incy);
  
// }

// template<>
// int deprecated_im2winConvCUDAimplentNHWC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     //set memory
//     // float *d_inptr, *d_filptr, *d_outptr;
//     size_t *d_offset_Y, *d_offset_D, *d_size_A, *d_size_outputHW;
//     //alloca memory
//     // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
//     // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
//     // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
//     cudaMalloc(&d_offset_Y, sizeof(size_t)*3);
//     cudaMalloc(&d_offset_D, sizeof(size_t)*3);
//     cudaMalloc(&d_size_A, sizeof(size_t)*2);
//     cudaMalloc(&d_size_outputHW, sizeof(size_t)*2);

//     size_t input_batch, input_channel, input_height, input_width,
//            filter_batch, filter_channel, filter_height, filter_width,
//            output_batch, output_channel, output_height, output_width,
//            stride_height, stride_width;

//     input_batch = dims_a[0];
//     input_channel = dims_a[1];
//     input_height = dims_a[2];
//     input_width = dims_a[3];

//     filter_batch = dims_b[0];
//     filter_channel = dims_b[1];
//     filter_height = dims_b[2];
//     filter_width = dims_b[3];

//     output_batch = dims_c[0];
//     output_channel = dims_c[1];
//     output_height = dims_c[2];
//     output_width = dims_c[3];

//     stride_height = (input_height - filter_height) / (output_height - 1);
//     stride_width = (input_width - filter_width) / (output_width - 1);

//     size_t size_A[2] = {filter_height * filter_width * input_channel, filter_batch};
//     size_t offset_D[3] = {filter_height * input_width * input_channel, stride_width * filter_height * input_channel, output_height * filter_height * input_width * input_channel};
//     size_t offset_Y[3] = {output_width * filter_batch, filter_batch, output_height * output_width * filter_batch};
//     size_t size_outputHW[2] = {output_height, output_width};
//     //init input and filter
//     // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
//     cudaMemcpy(d_size_outputHW, size_outputHW, sizeof(size_t)*2, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_size_A, size_A, sizeof(size_t)*2, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_offset_D, offset_D, sizeof(size_t)*3, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_offset_Y, offset_Y, sizeof(size_t)*3, cudaMemcpyHostToDevice);
//     //set blockSize and gridSize
//     const dim3 blockSize(32, 32);
//     const dim3 gridSize((dims_c[2] - 1)/blockSize.x + 1, (dims_c[3] - 1)/blockSize.y + 1, dims_c[0]);
//     //run kernelConv
//     // printf("dims_b[1] : %d\n", int(dims_b[1]));
//     // printf("dims_c[2] : %d\n", int(dims_c[2]));

//     im2winKernelConvNHWC<<<gridSize, blockSize>>>(inptr, filptr, outptr, d_size_outputHW, d_size_A, d_offset_D, d_offset_Y);
//     //result to host
//     //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
//     //clean device memory
//     cudaFree(d_size_outputHW);
//     cudaFree(d_size_A);
//     cudaFree(d_offset_D);
//     cudaFree(d_offset_Y);
//     return 0;
// }

template<>
int im2winConvCUDAimplentNHWCBLAS(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}