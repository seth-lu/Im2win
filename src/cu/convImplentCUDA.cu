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
    // printf("dims_b[1] : %d\n", int(dims_b[1]));32
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

// template <int BLOCK_SIZE_M,
//           int BLOCK_SIZE_K,
//           int BLOCK_SIZE_N,
//           int THREAD_SIZE_M,
//           int THREAD_SIZE_N>
// __global__ void implicitKernelConvNHWC_base(float *d_inptr, float *d_filptr, float *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int input_batch = d_dims_a[0];
//     const int input_channel = d_dims_a[1];
//     const int input_height = d_dims_a[2];
//     const int input_width = d_dims_a[3];

//     const int filter_batch = d_dims_b[0];
//     const int filter_channel = d_dims_b[1];
//     const int filter_height = d_dims_b[2];
//     const int filter_width = d_dims_b[3];

//     const int output_batch = d_dims_c[0];
//     const int output_channel = d_dims_c[1];
//     const int output_height = d_dims_c[2];
//     const int output_width = d_dims_c[3];

//     const int stride_height = (input_height - filter_height) / (output_height - 1);
//     const int stride_width = (input_width - filter_width) / (output_width - 1);

//     const int M = output_channel;
//     const int N = output_batch * output_height * output_width;
//     const int K = input_channel * filter_height * filter_width;

//     const int lda = K;
//     const int ldb = N;
//     const int ldc = N;

//     // d_inptr = d_inptr + bx * lda;
//     // d_filptr = d_filptr + by;
//     // d_outptr = d_outptr + bx * ldc + by;

//     for(int k_count = 0; k_count < K; ++k_count){
//         int id_oc = bx*blockDim.x + tx;

//         int id_ob = (by*blockDim.y + ty) / (output_height*output_width);
//         int res_ob = (by*blockDim.y + ty) % (output_height*output_width);
//         int id_oh = res_ob / output_width;
//         int id_ow = res_ob % output_width;

//         int id_ic = k_count / (filter_height*filter_width);
//         int res_ic = k_count % (filter_height*filter_width);
//         int id_fh = res_ic / filter_width;
//         int id_fw = res_ic % filter_width;

//         int id_ih = id_oh * stride_height + id_fh;
//         int id_iw = id_ow * stride_width + id_fw;

//         d_outptr[id_ob * output_height * output_width * output_channel 
//                  + id_oh * output_width * output_channel 
//                  + id_ow * output_channel
//                  + id_oc] += d_inptr[id_ob * input_height * input_width * input_channel
//                                      + id_ih * input_width * input_channel
//                                      + id_iw * input_channel
//                                      + id_ic] *
//                              d_filptr[id_oc * filter_height * filter_width * filter_channel
//                                      + id_fh * filter_width * filter_channel
//                                      + id_fw * filter_channel
//                                      + id_ic];
//     }
// }

// template<>
// void implicitConvCUDAimplentNHWC_base(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     //set memory
//     // float *d_inptr, *d_filptr, *d_outptr;
//     size_t *d_dims_a, *d_dims_b, *d_dims_c;
//     cudaMalloc(&d_dims_a, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_b, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_c, sizeof(size_t)*4);
//     //init input and filter
//     // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     //set blockSize and gridSize
//     const int M = dims_c[1]; //16
//     const int N = dims_c[0] * dims_c[2] * dims_c[3]; //64
//     const int K = dims_b[1] * dims_b[2] * dims_b[3]; //4
//     constexpr int BLOCK_SIZE_M = 16;
//     constexpr int BLOCK_SIZE_N = 16;
//     constexpr int BLOCK_SIZE_K = 16;
//     constexpr int THREAD_SIZE_M = 1;
//     constexpr int THREAD_SIZE_N = 1;
//     const dim3 blockSize(BLOCK_SIZE_M/THREAD_SIZE_M, BLOCK_SIZE_N/THREAD_SIZE_N);
//     const dim3 gridSize(CEIL_DIV(M, BLOCK_SIZE_M), 
//                         CEIL_DIV(N, BLOCK_SIZE_N));

//     implicitKernelConvNHWC_base<BLOCK_SIZE_M,
//                          BLOCK_SIZE_N,
//                          BLOCK_SIZE_K,
//                          THREAD_SIZE_M,
//                          THREAD_SIZE_N><<<gridSize, blockSize>>>
//                          (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);

//     //result to host
//     //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
//     //clean device memory
//     cudaFree(d_dims_a);
//     cudaFree(d_dims_b);
//     cudaFree(d_dims_c);
// }

template <int TLIE_M_PER_BLOCK,
          int TLIE_N_PER_BLOCK,
          int TLIE_K_PER_BLOCK,
          int TLIE_M_PER_THREAD,
          int TLIE_N_PER_THREAD>
__global__ void implicitKernelConvNHWC(float *d_inptr, float *d_filptr, float *d_outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
    const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
    const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;
    
    const int tid = threadIdx.x;
    const int tx = tid / THREAD_N_PER_BLOCK;
    const int ty = tid % THREAD_N_PER_BLOCK;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    //if(tid==0)printf("THREAD_PER_BLOCK :%d\n",THREAD_PER_BLOCK);
    const int input_batch = d_dims_a[0];
    const int input_channel = d_dims_a[1];
    const int input_height = d_dims_a[2];
    const int input_width = d_dims_a[3];

    const int filter_batch = d_dims_b[0];
    const int filter_channel = d_dims_b[1];
    const int filter_height = d_dims_b[2];
    const int filter_width = d_dims_b[3];

    const int output_batch = d_dims_c[0];
    const int output_channel = d_dims_c[1];
    const int output_height = d_dims_c[2];
    const int output_width = d_dims_c[3];

    const int stride_height = (input_height - filter_height) / (output_height - 1);
    const int stride_width = (input_width - filter_width) / (output_width - 1);

    const int M = output_channel;
    const int N = output_batch * output_height * output_width;
    const int K = input_channel * filter_height * filter_width;

    __shared__ float AS[TLIE_K_PER_BLOCK * TLIE_N_PER_BLOCK];
    __shared__ float BS[TLIE_K_PER_BLOCK * TLIE_M_PER_BLOCK];

    float reg_A[TLIE_N_PER_THREAD];
    float reg_B[TLIE_M_PER_THREAD];
    float reg_C[TLIE_M_PER_THREAD * TLIE_N_PER_THREAD];

    const int A_TILE_THREAD_PER_ROW = TLIE_N_PER_BLOCK;
    const int B_TILE_THREAD_PER_ROW = TLIE_M_PER_BLOCK;

    const int A_TILE_ROW = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;

    const int A_TILE_ROW_STRIDE = THREAD_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    #pragma unroll
    for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){
        //M = o_c and N = o_b * o_h * o_w
        //load A to shared mem
        #pragma unroll
        for (int i = 0; i < TLIE_K_PER_BLOCK; i += A_TILE_ROW_STRIDE){
            int id_ob = (by*TLIE_N_PER_BLOCK + A_TILE_COL) / (output_height*output_width);
            int res_ob = (by*TLIE_N_PER_BLOCK + A_TILE_COL) % (output_height*output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            int id_ic = (k_count+A_TILE_ROW+i) / (filter_height*filter_width);
            int res_ic = (k_count+A_TILE_ROW+i) % (filter_height*filter_width);
            int id_fh = res_ic / filter_width;
            int id_fw = res_ic % filter_width;

            int id_ih = id_oh * stride_height + id_fh;
            int id_iw = id_ow * stride_width + id_fw;
            
            AS[(A_TILE_ROW+i)*TLIE_N_PER_BLOCK+A_TILE_COL] =
            d_inptr[ id_ob * input_height * input_width * input_channel
                            + id_ih * input_width * input_channel
                            + id_iw * input_channel
                            + id_ic];
        }
        // if(tid==0){
        //     printf("AS :");
        //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
        //         for (int j = 0; j < TLIE_N_PER_BLOCK; ++j){
        //             printf(" %f",AS[i*TLIE_N_PER_BLOCK+j]);
        //         }
        //         printf("\n");
        //     }
        // }
        #pragma unroll
        for (int j = 0; j < TLIE_K_PER_BLOCK; j += B_TILE_ROW_STRIDE){
            int id_oc = bx*TLIE_M_PER_BLOCK + B_TILE_COL; 

            int id_ic = (k_count+B_TILE_ROW+j) / (filter_height*filter_width);
            int res_ic = (k_count+B_TILE_ROW+j) % (filter_height*filter_width);
            int id_fh = res_ic / filter_width;
            int id_fw = res_ic % filter_width;
            
            BS[(B_TILE_ROW+j)*TLIE_M_PER_BLOCK+B_TILE_COL] =
            d_filptr[id_oc * filter_height * filter_width * filter_channel
                     + id_fh * filter_width * filter_channel
                     + id_fw * filter_channel
                     + id_ic];
            //printf("BS_id : [%d,%d,%d,%d]\n",id_oc,id_ic,id_fh,id_fw);
        }
        __syncthreads();

        // if(tid==0){
        //     printf("BS :");
        //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
        //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
        //             printf(" %f",BS[i*TLIE_M_PER_BLOCK+j]);
        //         }
        //         printf("\n");
        //     }
        // }
        #pragma unroll
        for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){ 
            #pragma unroll
            for (int i = 0; i < TLIE_N_PER_THREAD; ++i){
                reg_A[i] = AS[inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+i];
            }
            #pragma unroll
            for (int j = 0; j < TLIE_M_PER_THREAD; ++j){
                reg_B[j] = BS[inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+j];
            }
            __syncthreads();
            // if(tid==1){
            //     printf("regA :");
            //     for (int i = 0; i < TLIE_N_PER_THREAD; ++i){
            //         printf(" %f",reg_A[i]);
            //     }
            //     printf("\n");
            // }

            // if(tid==1){
            //     printf("regB :");
            //     for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
            //         printf(" %f",reg_B[i]);
            //     }
            //     printf("\n");
            // }
            #pragma unroll
            for (int j = 0; j < TLIE_M_PER_THREAD; ++j){ 
                #pragma unroll
                for (int i = 0; i < TLIE_N_PER_THREAD; ++i){ 
                    reg_C[j*TLIE_N_PER_THREAD+i] += reg_A[i] * reg_B[j];
                }
            }
        } 
    }

    // if(tid==1){
    // printf("regC :");
    // for (int j = 0; j < TLIE_M_PER_THREAD; ++j)
    //     for (int i = 0; i < TLIE_N_PER_THREAD; ++i){
    //         printf(" %f",reg_C[j*TLIE_N_PER_THREAD+i]);
    //     }
    //     printf("\n");
    // }
    #pragma unroll
    for (int j = 0; j < TLIE_M_PER_THREAD; ++j){
        #pragma unroll
        for (int i = 0; i < TLIE_N_PER_THREAD; ++i){
            int id_oc = bx*TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + j; 

            int id_ob = (by*TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + i) / (output_height*output_width);
            int res_ob = (by*TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + i) % (output_height*output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            d_outptr[id_ob * output_height * output_width * output_channel 
                    + id_oh * output_width * output_channel 
                    + id_ow * output_channel
                    + id_oc] = reg_C[j*TLIE_N_PER_THREAD+i];
            // if(tid==2){
            //     printf("indexY :[%d,%d,%d,%d]",id_ob,id_oc,id_oh,id_ow);
            //     printf(" regC: %f\n",reg_C[j*TLIE_N_PER_THREAD+i]);
            // }
        }
    }
}

template<>
void implicitConvCUDAimplentNHWC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    //set memory
    // float *d_inptr, *d_filptr, *d_outptr;
    size_t *d_dims_a, *d_dims_b, *d_dims_c;
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
    const int M = dims_c[1]; //1
    const int N = dims_c[0] * dims_c[2] * dims_c[3]; //1 * 2 * 2
    const int K = dims_b[1] * dims_b[2] * dims_b[3]; //1 * 2 * 2
    constexpr int TLIE_M_PER_BLOCK = 64;
    constexpr int TLIE_N_PER_BLOCK = 128;
    constexpr int TLIE_K_PER_BLOCK = 16;
    constexpr int TLIE_M_PER_THREAD = 1;
    constexpr int TLIE_N_PER_THREAD = 1;
    const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
    const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
                        CEIL_DIV(N, TLIE_N_PER_BLOCK));

    implicitKernelConvNHWC< TLIE_M_PER_BLOCK,
                            TLIE_N_PER_BLOCK,
                            TLIE_K_PER_BLOCK,
                            TLIE_M_PER_THREAD,
                            TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
                            (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);

    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
void implicitConvCUDAimplentNHWC(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return;
}


template<>
void test_cuda<int>(int input){
    std::cout<<"run complete"<<std::endl;
    return;
}

template< int BLOCKSIZE >
__global__ void im2winKernelConvNCHWBASE(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    size_t tid, Idx_x, Idx_y, Idx_z, Idx, threadsPerBlock,
           input_batch, input_channel, input_height, input_width,
           filter_batch, filter_channel, filter_height, filter_width,
           output_batch, output_channel, output_height, output_width,
           stride_height, stride_width;
    
    threadsPerBlock = blockDim.x;
    tid = threadIdx.x;
    Idx_x = blockIdx.x * BLOCKSIZE + tid%BLOCKSIZE;
    Idx_y = blockIdx.y * BLOCKSIZE + tid/BLOCKSIZE;
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
    // __shared__ float smemB[12*1024];
    // int smemBsize = filter_channel * filter_height * filter_width;
    // int transIter = smemBsize / threadsPerBlock;
    // int resTransIter = smemBsize % threadsPerBlock;
    // float *to_filtptr = filptr + Idx_z * filter_channel * filter_height * filter_width;
    // for (int i = 0; i < transIter; ++i){
    //     smemB[i * threadsPerBlock + tid]
    //     = to_filtptr[i * threadsPerBlock + tid];
    // }
    // if (tid < resTransIter){
    //     smemB[transIter * threadsPerBlock + tid]
    //     = to_filtptr[transIter * threadsPerBlock + tid];
    // }
    // __syncthreads();
    if (Idx_x >= output_height || Idx_y >= output_width) return;
    
    size_t indexY = Idx_z * output_height * output_width
                    + Idx_x * output_width + Idx_y;
    float tmp = 0.0;
    
    for(size_t c = 0; c < input_channel; ++c){
        for(size_t w = 0; w < filter_width; ++w){
            for(size_t h = 0; h < filter_height; ++h){
                size_t indexD = c * output_height * filter_height * input_width
                                + w * filter_height + h
                                + Idx_x * filter_height * input_width
                                + Idx_y * stride_width * filter_height;
                size_t indexW = Idx_z * filter_channel * filter_height * filter_width
                                + c * filter_height * filter_width
                                + w * filter_height + h;
                tmp += inptr[indexD] * filptr[indexW];
            }
        }
    }
    outptr[indexY] = tmp;
}

template <>
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
    constexpr int BLOCKSIZE = 32;
    const dim3 blockSize(BLOCKSIZE * BLOCKSIZE);
    const dim3 gridSize((dims_c[2] - 1)/BLOCKSIZE + 1, (dims_c[3] - 1)/BLOCKSIZE + 1, dims_c[1]);
    //run kernelConv
    // printf("dims_b[1] : %d\n", int(dims_b[1]));
    // printf("dims_c[2] : %d\n", int(dims_c[2]));

    for(size_t i = 0; i < dims_a[0]; ++i)
        im2winKernelConvNCHWBASE<BLOCKSIZE>
                                <<<gridSize, blockSize>>>(inptr+i*dims_a[1]*dims_c[2]*dims_b[2]*dims_a[3], filptr, outptr+i*dims_c[1]*dims_c[2]*dims_c[3], d_dims_a, d_dims_b, d_dims_c);
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

// template< int TLIE_M_PER_BLOCK,
//           int TLIE_N_PER_BLOCK,
//           int TLIE_K_PER_BLOCK,
//           int TLIE_M_PER_THREAD,
//           int TLIE_N_PER_THREAD >
// __global__ void im2winConvCUDAimplentNCHWHPCkernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
//     const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
//     const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
//     const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

//     const int K = d_dims_b[1];

//     const int tid = threadIdx.x;
//     const int tx = tid / THREAD_N_PER_BLOCK;
//     const int ty = tid % THREAD_N_PER_BLOCK;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int input_batch = d_dims_a[0];
//     const int input_channel = d_dims_a[1];
//     const int input_height = d_dims_a[2];
//     const int input_width = d_dims_a[3];

//     const int filter_batch = d_dims_b[0];
//     const int filter_channel = d_dims_b[1];
//     const int filter_height = d_dims_b[2];
//     const int filter_width = d_dims_b[3];

//     const int output_batch = d_dims_c[0];
//     const int output_channel = d_dims_c[1];
//     const int output_height = d_dims_c[2];
//     const int output_width = d_dims_c[3];

//     const int stride_height = (input_height - filter_height) / (output_height - 1);
//     const int stride_width = (input_width - filter_width) / (output_width - 1);

//     __shared__ float smem[12 * 1024];
//     float *Asmem = reinterpret_cast<float *>(smem);
//     int Asmem_size = input_width * filter_height * TLIE_N_PER_BLOCK * TLIE_K_PER_BLOCK;
//     float *Bsmem = reinterpret_cast<float *>(smem + 6 * 1024);
//     int Bsmem_size = filter_height * filter_width * TLIE_M_PER_BLOCK * TLIE_K_PER_BLOCK;
//     float *Csmem = reinterpret_cast<float *>(smem + 11 * 1024);
//     int Csmem_size = TLIE_M_PER_BLOCK*TLIE_N_PER_BLOCK*output_width;
//     if(Asmem_size>6*1024||Bsmem_size>5*1024||Csmem_size>1024){
//         if((bx==0&&by==0)&&tid==0) printf("smem init error!\n");
//         return;
//     }
//     memset(Csmem, 0, sizeof(float)*Csmem_size);  

//     float Areg[TLIE_N_PER_THREAD];
//     float Breg[TLIE_M_PER_THREAD];
//     float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];

//     const int THREAD_TILE_A_PER_ROW = TLIE_N_PER_BLOCK * TLIE_K_PER_BLOCK;
//     const int THREAD_TILE_B_PER_ROW = TLIE_M_PER_BLOCK * TLIE_K_PER_BLOCK;

//     const int TILE_A_ROW = tid / THREAD_TILE_A_PER_ROW;
//     const int TILE_B_ROW = tid / THREAD_TILE_B_PER_ROW;

//     const int TILE_A_COL = tid % THREAD_TILE_A_PER_ROW;
//     const int TILE_B_COL = tid % THREAD_TILE_B_PER_ROW;

//     const int TILE_A_ROW_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_ROW;
//     const int TILE_B_ROW_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_ROW;

//     if(THREAD_PER_BLOCK % TILE_A_ROW_STRIDE !=0||THREAD_PER_BLOCK % TILE_B_ROW_STRIDE!=0){printf("TILE_THREAD_STRIDE ERROR\n");return;}
//     int sum_n = (by * TLIE_N_PER_BLOCK + TILE_A_ROW/TLIE_K_PER_BLOCK);
//     int id_ob_copy = (by * TLIE_N_PER_BLOCK + TILE_A_ROW/TLIE_K_PER_BLOCK)/(output_height);
//     int id_oh_copy = (by * TLIE_N_PER_BLOCK + TILE_A_ROW/TLIE_K_PER_BLOCK)%(output_height);
//     //printf("id_ob : %d, id_oh : %d, sum_n : %d, TILE_A_ROW : %d, by: %d\n", id_ob_copy, id_oh_copy, sum_n, TILE_A_ROW, by);    
//     #pragma unroll
//     for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){
//         // if(by==0&&tid==0){
//         //     printf("big K loop start, iter: %d\n",k_count);
//         // }
//         #pragma unroll
//         for (int i = 0; i < filter_height * filter_width; i += TILE_B_ROW_STRIDE){
//             int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL/TLIE_K_PER_BLOCK;
//             int id_ic = k_count + TILE_B_COL%TLIE_K_PER_BLOCK;
//             Bsmem[(TILE_B_ROW+i)*TLIE_M_PER_BLOCK * TLIE_K_PER_BLOCK+TILE_B_COL] = 
//             filptr[id_oc*filter_channel*filter_height*filter_width
//                   +id_ic*filter_height*filter_width
//                   +TILE_B_ROW+i];
//             // int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + TILE_B_COL;
//             // int id_ic = (k_count + TILE_B_ROW + i) / (filter_height*filter_width);
//             // int res_ic = (k_count + TILE_B_ROW + i) % (filter_height*filter_width);
//             // int id_fh = res_ic / filter_width;
//             // int id_fw = res_ic % filter_width;

//             // Bsmem[(TILE_B_ROW+i)*TLIE_M_PER_BLOCK+TILE_B_COL] = 
//             // filptr[id_oc*filter_channel*filter_height*filter_width
//             //       +id_ic*filter_height*filter_width
//             //       +id_fh
//             //       +id_fw*filter_height];
//         }
//         #pragma unroll
//         for (int j = 0; j < input_width*filter_height; j += TILE_A_ROW_STRIDE){
//             int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL/TLIE_K_PER_BLOCK)/(output_height);
//             int id_oh = (by * TLIE_N_PER_BLOCK + TILE_A_COL/TLIE_K_PER_BLOCK)%(output_height);
//             //printf("id_ob = %d, id_oh = %d\n", id_ob,id_oh);
//             int id_ic = k_count+TILE_A_COL%TLIE_K_PER_BLOCK;
//             Asmem[(TILE_A_ROW+j)*TLIE_N_PER_BLOCK*TLIE_K_PER_BLOCK+TILE_A_COL] = 
//             inptr[id_ob*input_channel*output_height*filter_height*input_width
//                   +id_ic*output_height*filter_height*input_width
//                   +id_oh*input_width*filter_height
//                   +TILE_A_ROW+j];
//             // printf("by = %d, tid = %d, TILE_A_COL = %d, input = %f\n",by, tid,TILE_A_COL,inptr[id_ob*input_channel*output_height*filter_height*filter_width
//             //       +id_ic*output_height*filter_height*input_width
//             //       +id_oh*input_width*filter_height
//             //       +TILE_A_ROW+j]);
//             // int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + TILE_A_COL)/(output_height * output_width);
//             // int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + TILE_A_COL)%(output_height * output_width);
//             // int id_oh = res_ob / output_width;
//             // int id_ow = res_ob % output_width;

//             // int id_ic = (k_count + TILE_A_ROW + j) / (filter_height*filter_width);
//             // int res_ic = (k_count + TILE_A_ROW + j) % (filter_height*filter_width);
//             // int id_fh = res_ic / filter_width;
//             // int id_fw = res_ic % filter_width;

//             // int id_ih = id_oh*filter_height*input_width+id_fh;
//             // int id_iw = id_ow*stride_width+id_fw;

//             // Asmem[(TILE_A_ROW+j)*TLIE_N_PER_BLOCK+TILE_A_COL] = 
//             // inptr[id_ob*input_channel*output_height*filter_height*filter_width
//             //      +id_ic*output_height*filter_height*input_width
//             //      +id_ih
//             //      +id_iw*filter_height];
//         }
//         // __syncthreads();

//         // if((bx==0&&by==0)&&tid==0){
//         //     printf("Asmem : ");
//         //     printf("input_width * filter_height :%d\n",input_width * filter_height);
//         //     for (int i = 0; i < input_width * filter_height; ++i){
//         //         for (int j = 0; j < TLIE_K_PER_BLOCK * TLIE_N_PER_BLOCK; ++j){
//         //             printf(" %f",Asmem[i*TLIE_K_PER_BLOCK * TLIE_N_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }

//         // if((bx==0&&by==0)&&tid==0){
//         //     printf("Bsmem :");
//         //     for (int i = 0; i < filter_height * filter_width; ++i){
//         //         for (int j = 0; j < TLIE_K_PER_BLOCK * TLIE_M_PER_BLOCK; ++j){
//         //             printf(" %f",Bsmem[i*TLIE_K_PER_BLOCK * TLIE_M_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }
//         #pragma unroll
//         for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){
//             #pragma unroll
//             for (int j = 0; j < output_width; ++j){
//                 memset(Creg, 0, sizeof(Creg));
//                 #pragma unroll
//                 for (int h = 0; h < filter_height; ++h){
//                     #pragma unroll
//                     for (int w = 0; w < filter_width; ++w){
//                         int B_offset = h * filter_width + w;
//                         int A_offset = B_offset + j * stride_width * filter_height;
//                         #pragma unroll
//                         for (int m = 0; m < TLIE_M_PER_THREAD; ++m)
//                         Breg[m] = Bsmem[B_offset*(TLIE_M_PER_BLOCK*TLIE_K_PER_BLOCK)+(inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD)+m];
//                         #pragma unroll
//                         for (int n = 0; n < TLIE_N_PER_THREAD; ++n)
//                         Areg[n] = Asmem[A_offset*(TLIE_N_PER_BLOCK*TLIE_K_PER_BLOCK)+(inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD)+n];
//                         #pragma unroll
//                         for (int m = 0; m < TLIE_M_PER_THREAD; ++m){
//                             #pragma unroll
//                             for (int n = 0; n < TLIE_N_PER_THREAD; ++n){
//                                 Creg[m*TLIE_N_PER_THREAD+n] +=
//                                 Breg[m] * Areg[n];                                      
//                             }   
//                         }

//                     }
//                 }
//                 #pragma unroll
//                 for (int m = 0; m < TLIE_M_PER_THREAD; ++m){
//                     #pragma unroll
//                     for (int n = 0; n < TLIE_N_PER_THREAD; ++n){
//                         Csmem[(ty*TLIE_N_PER_THREAD+n)*TLIE_M_PER_BLOCK*output_width+(tx*TLIE_M_PER_THREAD+m)*output_width+j] += Creg[m*TLIE_N_PER_THREAD+n];                                  
//                     }
//                 }
//             }
//         }
//     }

//     // if((bx==0&&by==0)&&tid==0){
//     //     printf("Csmem :");
//     //     for (int i = 0; i < TLIE_M_PER_BLOCK * TLIE_N_PER_BLOCK; ++i){
//     //         for (int j = 0; j < output_width; ++j){
//     //             printf(" %f",Csmem[i*output_width+j]);
//     //         }
//     //         printf("\n");
//     //     }
//     // }

//     #pragma unroll
//     for (int m = 0; m < TLIE_M_PER_THREAD; ++m){
//         #pragma unroll
//         for (int n = 0; n < TLIE_N_PER_THREAD; ++n){
//             #pragma unroll
//             for (int j = 0; j < output_width; ++j){
//                 int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + m;
//                 int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + n)/(output_height);
//                 int id_oh = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + n)%(output_height);
//                 int id_ow = j;

//                 outptr[id_ob*output_channel*output_height*output_width
//                 +id_oc*output_height*output_width
//                 +id_oh*output_width
//                 +id_ow] = Csmem[(ty*TLIE_N_PER_THREAD+n)*TLIE_M_PER_BLOCK*output_width+(tx*TLIE_M_PER_THREAD+m)*output_width+j];
//             }
//         }
//     }
//     //printf("Idx_x : %d\n", Idx_x);
//     // printf("Idx_y : %d\n", Idx_y);
// }

// template<>
// int im2winConvCUDAimplentNCHWHPC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     //set memory
//     // float *d_inptr, *d_filptr, *d_outptr;
//     size_t *d_dims_a, *d_dims_b, *d_dims_c;
//     //alloca memory
//     // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
//     // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
//     // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
//     cudaMalloc(&d_dims_a, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_b, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_c, sizeof(size_t)*4);
//     //init input and filter
//     // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     //set blockSize and gridSize
//     const int M = dims_c[1]; //1
//     const int N = dims_c[2] * dims_c[0]; //1 * 2 * 2
//     const int K = dims_b[1]; //1 * 2 * 2

//     constexpr int TLIE_M_PER_BLOCK = 16;
//     constexpr int TLIE_N_PER_BLOCK = 16;
//     constexpr int TLIE_K_PER_BLOCK = 1;
//     constexpr int TLIE_M_PER_THREAD = 1;
//     constexpr int TLIE_N_PER_THREAD = 1;
//     const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
//     const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
//                         CEIL_DIV(N, TLIE_N_PER_BLOCK));

//     im2winConvCUDAimplentNCHWHPCkernel< TLIE_M_PER_BLOCK,
//                                         TLIE_N_PER_BLOCK,
//                                         TLIE_K_PER_BLOCK,
//                                         TLIE_M_PER_THREAD,
//                                         TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
//                                     (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
//     //result to host
//     //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
//     //clean device memory
//     cudaFree(d_dims_a);
//     cudaFree(d_dims_b);
//     cudaFree(d_dims_c);
// }


// template< int TLIE_M_PER_BLOCK,
//           int TLIE_N_PER_BLOCK,
//           int TLIE_K_PER_BLOCK,
//           int TLIE_M_PER_THREAD,
//           int TLIE_N_PER_THREAD >
// __global__ void im2winConvCUDAimplentNCHWV2kernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
//     const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
//     const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
//     const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

//     const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

//     const int tid = threadIdx.x;
//     const int tx = tid / THREAD_N_PER_BLOCK;
//     const int ty = tid % THREAD_N_PER_BLOCK;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int input_batch = d_dims_a[0];
//     const int input_channel = d_dims_a[1];
//     const int input_height = d_dims_a[2];
//     const int input_width = d_dims_a[3];

//     const int filter_batch = d_dims_b[0];
//     const int filter_channel = d_dims_b[1];
//     const int filter_height = d_dims_b[2];
//     const int filter_width = d_dims_b[3];

//     const int output_batch = d_dims_c[0];
//     const int output_channel = d_dims_c[1];
//     const int output_height = d_dims_c[2];
//     const int output_width = d_dims_c[3];

//     const int stride_height = (input_height - filter_height) / (output_height - 1);
//     const int stride_width = (input_width - filter_width) / (output_width - 1);

//     __shared__ float Asmem[TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
//     __shared__ float Bsmem[TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

//     float Areg[TLIE_N_PER_THREAD];
//     float Breg[TLIE_M_PER_THREAD];
//     float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];
//     memset(Creg, 0, sizeof(Creg));

//     const int THREAD_TILE_A_PER_COL = TLIE_K_PER_BLOCK;
//     const int THREAD_TILE_B_PER_COL = TLIE_K_PER_BLOCK;

//     const int TILE_A_ROW = tid % THREAD_TILE_A_PER_COL;
//     const int TILE_B_ROW = tid % THREAD_TILE_B_PER_COL;

//     const int TILE_A_COL = tid / THREAD_TILE_A_PER_COL;
//     const int TILE_B_COL = tid / THREAD_TILE_B_PER_COL;

//     const int TILE_A_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_COL;
//     const int TILE_B_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_COL;

//     if(THREAD_PER_BLOCK % THREAD_TILE_A_PER_COL !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_COL!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

//     for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){
//         #pragma unroll
//         for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
//             int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
//             int id_ic = (k_count + TILE_B_ROW) / (filter_height*filter_width);
//             int res_ic = (k_count + TILE_B_ROW) % (filter_height*filter_width);
//             int id_fh = res_ic % filter_height;
//             int id_fw = res_ic / filter_height;

//             Bsmem[(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
//             filptr[id_oc*filter_channel*filter_height*filter_width
//                   +id_ic*filter_height*filter_width
//                   +id_fh
//                   +id_fw*filter_height];
//         }
//         // if(tid==0){
//         //     printf("Bsmem :");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
//         //             printf(" %f",Bsmem[i*TLIE_M_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }
//         #pragma unroll
//         for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
//             int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)/(output_height * output_width);
//             int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)%(output_height * output_width);
//             int id_oh = res_ob / output_width;
//             int id_ow = res_ob % output_width;

//             int id_ic = (k_count + TILE_A_ROW) / (filter_height*filter_width);
//             int res_ic = (k_count + TILE_A_ROW) % (filter_height*filter_width);
//             int id_fh = res_ic % filter_height;
//             int id_fw = res_ic / filter_height;

//             // int id_ih = id_oh*filter_height*input_width+id_fh;
//             // int id_iw = id_ow*stride_width+id_fw;
//             // if(tid==1) printf("input : %f, input_org :%f\n",inptr[id_ob*input_channel*output_height*filter_height*filter_width
//             //      +id_ic*output_height*filter_height*input_width
//             //      +id_oh*filter_height*input_width+id_fh
//             //      +id_ow*stride_width+id_fw],inptr[1]);
//             // printf("tid : %d, nchw: %d %d %d %d\n",tid, id_ob, id_ic, id_ih, id_iw);
//             Asmem[(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
//             inptr[id_ob*input_channel*output_height*filter_height*input_width
//                  +id_ic*output_height*filter_height*input_width
//                  +id_oh*filter_height*input_width
//                  +id_ow*stride_width*filter_height
//                  +id_fh
//                  +id_fw*filter_height];
//             // printf("asmem id : %d, inptr id : %d\n",(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL,id_ob*input_channel*output_height*filter_height*input_width
//             //      +id_ic*output_height*filter_height*input_width
//             //      +id_oh*filter_height*input_width
//             //      +id_ow*stride_width*filter_height
//             //      +id_fh
//             //      +id_fw*filter_height);
//             // printf("Asmem[1] : %f\n", Asmem[1]);
//             // if(tid==1) 
//             // printf("Asmem : %f, id :%d\n",Asmem[(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL],(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL);
//         }
//         // __syncthreads();
//         //     if((bx==0&&by==0)&&tid==0){
//         //     printf("Asmem : ");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_N_PER_BLOCK; ++j){
//         //             printf(" %f",Asmem[i*TLIE_N_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }

//         // if((bx==0&&by==0)&&tid==0){
//         //     printf("Bsmem :");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
//         //             printf(" %f",Bsmem[i*TLIE_M_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }
//         #pragma unroll
//         for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){
//             #pragma unroll
//             for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//                 Breg[i] = Bsmem[inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i];
//             }
//             #pragma unroll
//             for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//                 Areg[j] = Asmem[inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j];
//             }
//             #pragma unroll
//             for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//                 #pragma unroll
//                 for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//                     Creg[i*TLIE_N_PER_THREAD+j] +=
//                     Breg[i] * Areg[j];                                      
//                 }
//             }
//         }
//     }
    
//     #pragma unroll
//     for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//         #pragma unroll
//         for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//             int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
//             int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
//             int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
//             int id_oh = res_ob / output_width;
//             int id_ow = res_ob % output_width;

//             outptr[id_ob*output_channel*output_height*output_width
//             +id_oc*output_height*output_width
//             +id_oh*output_width
//             +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
//         }
//     }
//     //printf("Idx_x : %d\n", Idx_x);
//     // printf("Idx_y : %d\n", Idx_y);
// }

// template<>
// int im2winConvCUDAimplentNCHWV2(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     //set memory
//     // float *d_inptr, *d_filptr, *d_outptr;
//     size_t *d_dims_a, *d_dims_b, *d_dims_c;
//     //alloca memory
//     // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
//     // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
//     // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
//     cudaMalloc(&d_dims_a, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_b, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_c, sizeof(size_t)*4);
//     //init input and filter
//     // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);

//     //set blockSize and gridSize
//     int output_batch = dims_c[0];
//     int output_channel = dims_c[1];
//     int output_height = dims_c[2];
//     int output_width = dims_c[3];

//     int filter_batch = dims_b[0];
//     int filter_channel = dims_b[1];
//     int filter_height = dims_b[2];
//     int filter_width = dims_b[3];
//     if (filter_channel < 8) filter_channel = 8;

//     const int M = output_channel;
//     const int N = output_batch * output_height * output_width;
//     const int K = filter_channel * filter_height * filter_width;
//     constexpr int TLIE_M_PER_BLOCK = 128;
//     constexpr int TLIE_N_PER_BLOCK = 128;
//     constexpr int TLIE_K_PER_BLOCK = 8;
//     constexpr int TLIE_M_PER_THREAD = 8;
//     constexpr int TLIE_N_PER_THREAD = 8;
//     const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
//     const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
//                         CEIL_DIV(N, TLIE_N_PER_BLOCK));

//     im2winConvCUDAimplentNCHWV2kernel< TLIE_M_PER_BLOCK,
//                                         TLIE_N_PER_BLOCK,
//                                         TLIE_K_PER_BLOCK,
//                                         TLIE_M_PER_THREAD,
//                                         TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
//                             (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
//     //result to host
//     //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
//     //clean device memory
//     cudaFree(d_dims_a);
//     cudaFree(d_dims_b);
//     cudaFree(d_dims_c);
// }

// template<>
// int im2winConvCUDAimplentNCHWV2(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     return 0;
// }

// template< int TLIE_M_PER_BLOCK,
//           int TLIE_N_PER_BLOCK,
//           int TLIE_K_PER_BLOCK,
//           int TLIE_M_PER_THREAD,
//           int TLIE_N_PER_THREAD >
// __global__ void im2winConvCUDAimplentNCHWV3kernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
//     const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
//     const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
//     const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

//     const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

//     const int tid = threadIdx.x;
//     const int tx = tid / THREAD_N_PER_BLOCK;
//     const int ty = tid % THREAD_N_PER_BLOCK;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int input_batch = d_dims_a[0];
//     const int input_channel = d_dims_a[1];
//     const int input_height = d_dims_a[2];
//     const int input_width = d_dims_a[3];

//     const int filter_batch = d_dims_b[0];
//     const int filter_channel = d_dims_b[1];
//     const int filter_height = d_dims_b[2];
//     const int filter_width = d_dims_b[3];

//     const int output_batch = d_dims_c[0];
//     const int output_channel = d_dims_c[1];
//     const int output_height = d_dims_c[2];
//     const int output_width = d_dims_c[3];

//     const int stride_height = (input_height - filter_height) / (output_height - 1);
//     const int stride_width = (input_width - filter_width) / (output_width - 1);

//     __shared__ float Asmem[TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
//     __shared__ float Bsmem[TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

//     float Areg[TLIE_N_PER_THREAD];
//     float Breg[TLIE_M_PER_THREAD];
//     float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];
//     memset(Creg, 0, sizeof(Creg));

//     const int THREAD_TILE_A_PER_COL = TLIE_K_PER_BLOCK;
//     const int THREAD_TILE_B_PER_COL = TLIE_K_PER_BLOCK;

//     const int TILE_A_ROW = tid % THREAD_TILE_A_PER_COL;
//     const int TILE_B_ROW = tid % THREAD_TILE_B_PER_COL;

//     const int TILE_A_COL = tid / THREAD_TILE_A_PER_COL;
//     const int TILE_B_COL = tid / THREAD_TILE_B_PER_COL;

//     const int TILE_A_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_COL;
//     const int TILE_B_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_COL;

//     if(THREAD_PER_BLOCK % THREAD_TILE_A_PER_COL !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_COL!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

//     for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){
//         #pragma unroll
//         for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
//             int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
//             int id_ic = (k_count + TILE_B_ROW) / (filter_height*filter_width);
//             int res_ic = (k_count + TILE_B_ROW) % (filter_height*filter_width);
//             int id_fh = res_ic % filter_height;
//             int id_fw = res_ic / filter_height;

//             Bsmem[(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
//             filptr[id_oc*filter_channel*filter_height*filter_width
//                   +id_ic*filter_height*filter_width
//                   +id_fh
//                   +id_fw*filter_height];
//         }
//         // if(tid==0){
//         //     printf("Bsmem :");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
//         //             printf(" %f",Bsmem[i*TLIE_M_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }
//         #pragma unroll
//         for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
//             int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)/(output_height * output_width);
//             int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)%(output_height * output_width);
//             int id_oh = res_ob / output_width;
//             int id_ow = res_ob % output_width;

//             int id_ic = (k_count + TILE_A_ROW) / (filter_height*filter_width);
//             int res_ic = (k_count + TILE_A_ROW) % (filter_height*filter_width);
//             int id_fh = res_ic % filter_height;
//             int id_fw = res_ic / filter_height;

//             // int id_ih = id_oh*filter_height*input_width+id_fh;
//             // int id_iw = id_ow*stride_width+id_fw;
//             // if(tid==1) printf("input : %f, input_org :%f\n",inptr[id_ob*input_channel*output_height*filter_height*filter_width
//             //      +id_ic*output_height*filter_height*input_width
//             //      +id_oh*filter_height*input_width+id_fh
//             //      +id_ow*stride_width+id_fw],inptr[1]);
//             // printf("tid : %d, nchw: %d %d %d %d\n",tid, id_ob, id_ic, id_ih, id_iw);
//             Asmem[(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
//             inptr[id_ob*input_channel*output_height*filter_height*input_width
//                  +id_ic*output_height*filter_height*input_width
//                  +id_oh*filter_height*input_width
//                  +id_ow*stride_width*filter_height
//                  +id_fh
//                  +id_fw*filter_height];
//             // printf("asmem id : %d, inptr id : %d\n",(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL,id_ob*input_channel*output_height*filter_height*input_width
//             //      +id_ic*output_height*filter_height*input_width
//             //      +id_oh*filter_height*input_width
//             //      +id_ow*stride_width*filter_height
//             //      +id_fh
//             //      +id_fw*filter_height);
//             // printf("Asmem[1] : %f\n", Asmem[1]);
//             // if(tid==1) 
//             // printf("Asmem : %f, id :%d\n",Asmem[(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL],(TILE_A_ROW + j)*TLIE_K_PER_BLOCK+TILE_A_COL);
//         }
//         // __syncthreads();
//         //     if((bx==0&&by==0)&&tid==0){
//         //     printf("Asmem : ");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_N_PER_BLOCK; ++j){
//         //             printf(" %f",Asmem[i*TLIE_N_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }

//         // if((bx==0&&by==0)&&tid==0){
//         //     printf("Bsmem :");
//         //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
//         //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
//         //             printf(" %f",Bsmem[i*TLIE_M_PER_BLOCK+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         // }
//         #pragma unroll
//         for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){
//             #pragma unroll
//             for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//                 Breg[i] = Bsmem[inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i];
//             }
//             #pragma unroll
//             for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//                 Areg[j] = Asmem[inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j];
//             }
//             #pragma unroll
//             for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//                 #pragma unroll
//                 for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//                     Creg[i*TLIE_N_PER_THREAD+j] +=
//                     Breg[i] * Areg[j];                                      
//                 }
//             }
//         }
//     }
    
//     #pragma unroll
//     for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
//         #pragma unroll
//         for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
//             int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
//             int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
//             int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
//             int id_oh = res_ob / output_width;
//             int id_ow = res_ob % output_width;

//             outptr[id_ob*output_channel*output_height*output_width
//             +id_oc*output_height*output_width
//             +id_oh*output_width
//             +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
//         }
//     }
//     //printf("Idx_x : %d\n", Idx_x);
//     // printf("Idx_y : %d\n", Idx_y);
// }

// template<>
// int im2winConvCUDAimplentNCHWV3(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     //set memory
//     // float *d_inptr, *d_filptr, *d_outptr;
//     size_t *d_dims_a, *d_dims_b, *d_dims_c;
//     //alloca memory
//     // cudaMalloc(&d_inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3]);
//     // cudaMalloc(&d_filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3]);
//     // cudaMalloc(&d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3]);
//     cudaMalloc(&d_dims_a, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_b, sizeof(size_t)*4);
//     cudaMalloc(&d_dims_c, sizeof(size_t)*4);
//     //init input and filter
//     // cudaMemcpy(d_inptr, inptr, sizeof(float)*dims_a[0]*dims_a[1]*dims_a[2]*dims_a[3], cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_filptr, filptr, sizeof(float)*dims_b[0]*dims_b[1]*dims_b[2]*dims_b[3], cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_a, dims_a, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_b, dims_b, sizeof(size_t)*4, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims_c, dims_c, sizeof(size_t)*4, cudaMemcpyHostToDevice);

//     //set blockSize and gridSize
//     int output_batch = dims_c[0];
//     int output_channel = dims_c[1];
//     int output_height = dims_c[2];
//     int output_width = dims_c[3];

//     int filter_batch = dims_b[0];
//     int filter_channel = dims_b[1];
//     int filter_height = dims_b[2];
//     int filter_width = dims_b[3];
//     if (filter_channel < 8) filter_channel = 8;

//     const int M = output_channel;
//     const int N = output_batch * output_height * output_width;
//     const int K = filter_channel * filter_height * filter_width;
//     constexpr int TLIE_M_PER_BLOCK = 128;
//     constexpr int TLIE_N_PER_BLOCK = 128;
//     constexpr int TLIE_K_PER_BLOCK = 8;
//     constexpr int TLIE_M_PER_THREAD = 8;
//     constexpr int TLIE_N_PER_THREAD = 8;
//     const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
//     const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
//                         CEIL_DIV(N, TLIE_N_PER_BLOCK));

//     im2winConvCUDAimplentNCHWV3kernel< TLIE_M_PER_BLOCK,
//                                         TLIE_N_PER_BLOCK,
//                                         TLIE_K_PER_BLOCK,
//                                         TLIE_M_PER_THREAD,
//                                         TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
//                             (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
//     //result to host
//     //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
//     //clean device memory
//     cudaFree(d_dims_a);
//     cudaFree(d_dims_b);
//     cudaFree(d_dims_c);
// }

// template<>
// int im2winConvCUDAimplentNCHWV3(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     return 0;
// }

template< int TLIE_M_PER_BLOCK,
          int TLIE_N_PER_BLOCK,
          int TLIE_K_PER_BLOCK,
          int TLIE_M_PER_THREAD,
          int TLIE_N_PER_THREAD >
__global__ void im2winConvCUDAimplentNCHWHPC_vectorizedkernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
    const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
    const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

    const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

    const int tid = threadIdx.x;
    const int tx = tid / THREAD_N_PER_BLOCK;
    const int ty = tid % THREAD_N_PER_BLOCK;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int input_batch = d_dims_a[0];
    const int input_channel = d_dims_a[1];
    const int input_height = d_dims_a[2];
    const int input_width = d_dims_a[3];

    const int filter_batch = d_dims_b[0];
    const int filter_channel = d_dims_b[1];
    const int filter_height = d_dims_b[2];
    const int filter_width = d_dims_b[3];

    const int output_batch = d_dims_c[0];
    const int output_channel = d_dims_c[1];
    const int output_height = d_dims_c[2];
    const int output_width = d_dims_c[3];

    const int stride_height = (input_height - filter_height) / (output_height - 1);
    const int stride_width = (input_width - filter_width) / (output_width - 1);

    __shared__ float Asmem[2][TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
    __shared__ float Bsmem[2][TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

    float Areg[2][TLIE_N_PER_THREAD];
    float Breg[2][TLIE_M_PER_THREAD];
    float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];
    memset(Creg, 0, sizeof(Creg));

    const int THREAD_TILE_A_PER_COL = TLIE_K_PER_BLOCK;
    const int THREAD_TILE_B_PER_COL = TLIE_K_PER_BLOCK;

    const int TILE_A_ROW = tid % THREAD_TILE_A_PER_COL;
    const int TILE_B_ROW = tid % THREAD_TILE_B_PER_COL;

    const int TILE_A_COL = tid / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL = tid / THREAD_TILE_B_PER_COL;

    const int TILE_A_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_COL;

    if (THREAD_PER_BLOCK % THREAD_TILE_A_PER_COL !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_COL!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
        int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
        int id_ic = (0 + TILE_B_ROW) / (filter_height*filter_width);
        int res_ic = (0 + TILE_B_ROW) % (filter_height*filter_width);
        int id_fh = res_ic % filter_height;
        int id_fw = res_ic / filter_height;

        Bsmem[0][(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
        filptr[id_oc*filter_channel*filter_height*filter_width
                +id_ic*filter_height*filter_width
                +id_fh
                +id_fw*filter_height];
    }

    #pragma unroll
    for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
        int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)/(output_height * output_width);
        int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)%(output_height * output_width);
        int id_oh = res_ob / output_width;
        int id_ow = res_ob % output_width;

        int id_ic = (0 + TILE_A_ROW) / (filter_height*filter_width);
        int res_ic = (0 + TILE_A_ROW) % (filter_height*filter_width);
        int id_fh = res_ic % filter_height;
        int id_fw = res_ic / filter_height;

        Asmem[0][(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
        inptr[id_ob*input_channel*output_height*filter_height*input_width
                +id_ic*output_height*filter_height*input_width
                +id_oh*filter_height*input_width
                +id_ow*stride_width*filter_height
                +id_fh
                +id_fw*filter_height];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
        Breg[0][i] = Bsmem[0][0*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i];
    }
    #pragma unroll
    for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
        Areg[0][j] = Asmem[0][0*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j];
    }
    __syncthreads();
    int write_stage_idx = 1;
    int k_count = 0;
    do {
        k_count += TLIE_K_PER_BLOCK;
        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK - 1; ++inner_k){
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                Breg[(inner_k+1)%2][i] = Bsmem[load_stage_idx][(inner_k+1)*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i];
            }
            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                Areg[(inner_k+1)%2][j] = Asmem[load_stage_idx][(inner_k+1)*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j];
            }
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                #pragma unroll
                for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                    Creg[i*TLIE_N_PER_THREAD+j] +=
                    Breg[inner_k%2][i] * Areg[inner_k%2][j];                                      
                }
            }
        }

        if (k_count < K){
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
                int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
                int id_ic = (k_count + TILE_B_ROW) / (filter_height*filter_width);
                int res_ic = (k_count + TILE_B_ROW) % (filter_height*filter_width);
                int id_fh = res_ic % filter_height;
                int id_fw = res_ic / filter_height;

                Bsmem[write_stage_idx][(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
                filptr[id_oc*filter_channel*filter_height*filter_width
                    +id_ic*filter_height*filter_width
                    +id_fh
                    +id_fw*filter_height];
            }

            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
                int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)/(output_height * output_width);
                int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)%(output_height * output_width);
                int id_oh = res_ob / output_width;
                int id_ow = res_ob % output_width;

                int id_ic = (k_count + TILE_A_ROW) / (filter_height * filter_width);
                int res_ic = (k_count + TILE_A_ROW) % (filter_height * filter_width);
                int id_fh = res_ic % filter_height;
                int id_fw = res_ic / filter_height;

                Asmem[write_stage_idx][(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
                inptr[id_ob*input_channel*output_height*filter_height*input_width
                    +id_ic*output_height*filter_height*input_width
                    +id_oh*filter_height*input_width
                    +id_ow*stride_width*filter_height
                    +id_fh
                    +id_fw*filter_height];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

        #pragma unroll
        for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
            Breg[0][i] = Bsmem[load_stage_idx^1][0+tx*TLIE_M_PER_THREAD+i];
        }
        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
            Areg[0][j] = Asmem[load_stage_idx^1][0+ty*TLIE_N_PER_THREAD+j];
        }

        #pragma unroll
        for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                Creg[i*TLIE_N_PER_THREAD+j] +=
                Breg[1][i] * Areg[1][j];                                      
            }
        }
    }while (k_count < K);
    
    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
            int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
            int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            outptr[id_ob*output_channel*output_height*output_width
            +id_oc*output_height*output_width
            +id_oh*output_width
            +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
        }
    }
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
}

template<>
int im2winConvCUDAimplentNCHWHPC_vectorized(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
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
    int output_batch = dims_c[0];
    int output_channel = dims_c[1];
    int output_height = dims_c[2];
    int output_width = dims_c[3];

    int filter_batch = dims_b[0];
    int filter_channel = dims_b[1];
    int filter_height = dims_b[2];
    int filter_width = dims_b[3];
    if (filter_channel < 8) filter_channel = 8;

    const int M = output_channel;
    const int N = output_batch * output_height * output_width;
    const int K = filter_channel * filter_height * filter_width;
    constexpr int TLIE_M_PER_BLOCK = 128;
    constexpr int TLIE_N_PER_BLOCK = 128;
    constexpr int TLIE_K_PER_BLOCK = 8;
    constexpr int TLIE_M_PER_THREAD = 8;
    constexpr int TLIE_N_PER_THREAD = 8;
    const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
    const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
                        CEIL_DIV(N, TLIE_N_PER_BLOCK));

    im2winConvCUDAimplentNCHWHPC_vectorizedkernel< TLIE_M_PER_BLOCK,
                                        TLIE_N_PER_BLOCK,
                                        TLIE_K_PER_BLOCK,
                                        TLIE_M_PER_THREAD,
                                        TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
                            (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
int im2winConvCUDAimplentNCHWHPC_vectorized(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

template< int TLIE_M_PER_BLOCK,
          int TLIE_N_PER_BLOCK,
          int TLIE_K_PER_BLOCK,
          int TLIE_M_PER_THREAD,
          int TLIE_N_PER_THREAD >
__global__ void im2winConvCUDAimplentNCHWHPC_prefetchkernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
    const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
    const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

    const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

    const int tid = threadIdx.x;
    const int tx = tid / THREAD_N_PER_BLOCK;
    const int ty = tid % THREAD_N_PER_BLOCK;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int input_batch = d_dims_a[0];
    const int input_channel = d_dims_a[1];
    const int input_height = d_dims_a[2];
    const int input_width = d_dims_a[3];

    const int filter_batch = d_dims_b[0];
    const int filter_channel = d_dims_b[1];
    const int filter_height = d_dims_b[2];
    const int filter_width = d_dims_b[3];

    const int output_batch = d_dims_c[0];
    const int output_channel = d_dims_c[1];
    const int output_height = d_dims_c[2];
    const int output_width = d_dims_c[3];

    const int stride_height = (input_height - filter_height) / (output_height - 1);
    const int stride_width = (input_width - filter_width) / (output_width - 1);

    __shared__ float Asmem[TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
    __shared__ float Bsmem[TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

    float Areg[TLIE_N_PER_THREAD];
    float Breg[TLIE_M_PER_THREAD];
    float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];
    memset(Creg, 0, sizeof(Creg));

    const int THREAD_TILE_A_PER_COL = TLIE_K_PER_BLOCK;
    const int THREAD_TILE_B_PER_COL = TLIE_K_PER_BLOCK;

    const int TILE_A_ROW = tid % THREAD_TILE_A_PER_COL;
    const int TILE_B_ROW = tid % THREAD_TILE_B_PER_COL;

    const int TILE_A_COL = tid / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL = tid / THREAD_TILE_B_PER_COL;

    const int TILE_A_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_COL;

    if (THREAD_PER_BLOCK % THREAD_TILE_A_PER_COL !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_COL!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

    for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){

        #pragma unroll
        for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
            int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
            int id_ic = (k_count + TILE_B_ROW) / (filter_height*filter_width);
            int res_ic = (k_count + TILE_B_ROW) % (filter_height*filter_width);
            int id_fh = res_ic % filter_height;
            int id_fw = res_ic / filter_height;

            Bsmem[(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
            filptr[id_oc*filter_channel*filter_height*filter_width
                +id_ic*filter_height*filter_width
                +id_fh
                +id_fw*filter_height];
        }

        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
            int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            int id_ic = (k_count + TILE_A_ROW) / (filter_height * filter_width);
            int res_ic = (k_count + TILE_A_ROW) % (filter_height * filter_width);
            int id_fh = res_ic % filter_height;
            int id_fw = res_ic / filter_height;

            Asmem[(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
            inptr[id_ob*input_channel*output_height*filter_height*input_width
                +id_ic*output_height*filter_height*input_width
                +id_oh*filter_height*input_width
                +id_ow*stride_width*filter_height
                +id_fh
                +id_fw*filter_height];
        }
        __syncthreads();

        #pragma unroll
        for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; i += 4){
                FETCH_FLOAT4(Breg[i]) = FETCH_FLOAT4(Bsmem[inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i]);
            }
            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_THREAD; j += 4){
                FETCH_FLOAT4(Areg[j]) = FETCH_FLOAT4(Asmem[inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j]);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                #pragma unroll
                for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                    Creg[i*TLIE_N_PER_THREAD+j] +=
                    Breg[i] * Areg[j];                                      
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
            int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
            int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            outptr[id_ob*output_channel*output_height*output_width
            +id_oc*output_height*output_width
            +id_oh*output_width
            +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
        }
    }
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
}

template<>
int im2winConvCUDAimplentNCHWHPC_prefetch(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
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
    int output_batch = dims_c[0];
    int output_channel = dims_c[1];
    int output_height = dims_c[2];
    int output_width = dims_c[3];

    int filter_batch = dims_b[0];
    int filter_channel = dims_b[1];
    int filter_height = dims_b[2];
    int filter_width = dims_b[3];
    if (filter_channel < 8) filter_channel = 8;

    const int M = output_channel;
    const int N = output_batch * output_height * output_width;
    const int K = filter_channel * filter_height * filter_width;
    constexpr int TLIE_M_PER_BLOCK = 128;
    constexpr int TLIE_N_PER_BLOCK = 128;
    constexpr int TLIE_K_PER_BLOCK = 8;
    constexpr int TLIE_M_PER_THREAD = 8;
    constexpr int TLIE_N_PER_THREAD = 8;
    const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
    const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
                        CEIL_DIV(N, TLIE_N_PER_BLOCK));

    im2winConvCUDAimplentNCHWHPC_prefetchkernel< TLIE_M_PER_BLOCK,
                                        TLIE_N_PER_BLOCK,
                                        TLIE_K_PER_BLOCK,
                                        TLIE_M_PER_THREAD,
                                        TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
                            (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
int im2winConvCUDAimplentNCHWHPC_prefetch(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

template< int TLIE_M_PER_BLOCK,
          int TLIE_N_PER_BLOCK,
          int TLIE_K_PER_BLOCK,
          int TLIE_M_PER_THREAD,
          int TLIE_N_PER_THREAD >
__global__ void im2winConvCUDAimplentNCHWHPCkernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
    const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
    const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

    const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

    const int tid = threadIdx.x;
    const int tx = tid / THREAD_N_PER_BLOCK;
    const int ty = tid % THREAD_N_PER_BLOCK;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int input_batch = d_dims_a[0];
    const int input_channel = d_dims_a[1];
    const int input_height = d_dims_a[2];
    const int input_width = d_dims_a[3];

    const int filter_batch = d_dims_b[0];
    const int filter_channel = d_dims_b[1];
    const int filter_height = d_dims_b[2];
    const int filter_width = d_dims_b[3];

    const int output_batch = d_dims_c[0];
    const int output_channel = d_dims_c[1];
    const int output_height = d_dims_c[2];
    const int output_width = d_dims_c[3];

    const int stride_height = (input_height - filter_height) / (output_height - 1);
    const int stride_width = (input_width - filter_width) / (output_width - 1);

    __shared__ float Asmem[2][TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
    __shared__ float Bsmem[2][TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

    float Areg[2][TLIE_N_PER_THREAD];
    float Breg[2][TLIE_M_PER_THREAD];
    float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];
    memset(Creg, 0, sizeof(Creg));

    const int THREAD_TILE_A_PER_COL = TLIE_K_PER_BLOCK;
    const int THREAD_TILE_B_PER_COL = TLIE_K_PER_BLOCK;

    const int TILE_A_ROW = tid % THREAD_TILE_A_PER_COL;
    const int TILE_B_ROW = tid % THREAD_TILE_B_PER_COL;

    const int TILE_A_COL = tid / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL = tid / THREAD_TILE_B_PER_COL;

    const int TILE_A_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_COL;
    const int TILE_B_COL_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_COL;

    if (THREAD_PER_BLOCK % THREAD_TILE_A_PER_COL !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_COL!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
        int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
        int id_ic = (0 + TILE_B_ROW) / (filter_height*filter_width);
        int res_ic = (0 + TILE_B_ROW) % (filter_height*filter_width);
        int id_fh = res_ic % filter_height;
        int id_fw = res_ic / filter_height;

        Bsmem[0][(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
        filptr[id_oc*filter_channel*filter_height*filter_width
                +id_ic*filter_height*filter_width
                +id_fh
                +id_fw*filter_height];
    }

    #pragma unroll
    for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
        int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)/(output_height * output_width);
        int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL+j)%(output_height * output_width);
        int id_oh = res_ob / output_width;
        int id_ow = res_ob % output_width;

        int id_ic = (0 + TILE_A_ROW) / (filter_height*filter_width);
        int res_ic = (0 + TILE_A_ROW) % (filter_height*filter_width);
        int id_fh = res_ic % filter_height;
        int id_fw = res_ic / filter_height;

        Asmem[0][(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
        inptr[id_ob*input_channel*output_height*filter_height*input_width
                +id_ic*output_height*filter_height*input_width
                +id_oh*filter_height*input_width
                +id_ow*stride_width*filter_height
                +id_fh
                +id_fw*filter_height];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_THREAD; i += 4){
        FETCH_FLOAT4(Breg[0][i]) = FETCH_FLOAT4(Bsmem[0][0*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i]);
    }
    #pragma unroll
    for (int j = 0; j < TLIE_N_PER_THREAD; j += 4){
        FETCH_FLOAT4(Areg[0][j]) = FETCH_FLOAT4(Asmem[0][0*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j]);
    }

    int write_stage_idx = 1;
    int k_count = 0;
    do {
        k_count += TLIE_K_PER_BLOCK;
        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK - 1; ++inner_k){
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; i += 4){
                FETCH_FLOAT4(Breg[(inner_k+1)%2][i]) = FETCH_FLOAT4(Bsmem[load_stage_idx][(inner_k+1)*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i]);
            }
            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_THREAD; j += 4){
                FETCH_FLOAT4(Areg[(inner_k+1)%2][j]) = FETCH_FLOAT4(Asmem[load_stage_idx][(inner_k+1)*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j]);
            }
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                #pragma unroll
                for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                    Creg[i*TLIE_N_PER_THREAD+j] +=
                    Breg[inner_k%2][i] * Areg[inner_k%2][j];                                      
                }
            }
        }

        if (k_count < K){
            #pragma unroll
            for (int i = 0; i < TLIE_M_PER_BLOCK; i += TILE_B_COL_STRIDE){
                int id_oc = bx * TLIE_M_PER_BLOCK + TILE_B_COL + i;
                int id_ic = (k_count + TILE_B_ROW) / (filter_height*filter_width);
                int res_ic = (k_count + TILE_B_ROW) % (filter_height*filter_width);
                int id_fh = res_ic % filter_height;
                int id_fw = res_ic / filter_height;

                Bsmem[write_stage_idx][(TILE_B_ROW)*TLIE_M_PER_BLOCK+TILE_B_COL+i] = 
                filptr[id_oc*filter_channel*filter_height*filter_width
                    +id_ic*filter_height*filter_width
                    +id_fh
                    +id_fw*filter_height];
            }

            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_BLOCK; j += TILE_A_COL_STRIDE){
                int id_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)/(output_height * output_width);
                int res_ob = (by * TLIE_N_PER_BLOCK + TILE_A_COL + j)%(output_height * output_width);
                int id_oh = res_ob / output_width;
                int id_ow = res_ob % output_width;

                int id_ic = (k_count + TILE_A_ROW) / (filter_height * filter_width);
                int res_ic = (k_count + TILE_A_ROW) % (filter_height * filter_width);
                int id_fh = res_ic % filter_height;
                int id_fw = res_ic / filter_height;

                Asmem[write_stage_idx][(TILE_A_ROW)*TLIE_N_PER_BLOCK+TILE_A_COL+j] = 
                inptr[id_ob*input_channel*output_height*filter_height*input_width
                    +id_ic*output_height*filter_height*input_width
                    +id_oh*filter_height*input_width
                    +id_ow*stride_width*filter_height
                    +id_fh
                    +id_fw*filter_height];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

        #pragma unroll
        for (int i = 0; i < TLIE_M_PER_THREAD; i += 4){
            FETCH_FLOAT4(Breg[0][i]) = FETCH_FLOAT4(Bsmem[load_stage_idx^1][0+tx*TLIE_M_PER_THREAD+i]);
        }
        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_THREAD; j += 4){
            FETCH_FLOAT4(Areg[0][j]) = FETCH_FLOAT4(Asmem[load_stage_idx^1][0+ty*TLIE_N_PER_THREAD+j]);
        }

        #pragma unroll
        for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
            #pragma unroll
            for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                Creg[i*TLIE_N_PER_THREAD+j] +=
                Breg[1][i] * Areg[1][j];                                      
            }
        }
    }while (k_count < K);
    
    #pragma unroll
    for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
        #pragma unroll
        for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
            int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
            int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            outptr[id_ob*output_channel*output_height*output_width
            +id_oc*output_height*output_width
            +id_oh*output_width
            +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
        }
    }
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
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
    int output_batch = dims_c[0];
    int output_channel = dims_c[1];
    int output_height = dims_c[2];
    int output_width = dims_c[3];

    int filter_batch = dims_b[0];
    int filter_channel = dims_b[1];
    int filter_height = dims_b[2];
    int filter_width = dims_b[3];
    if (filter_channel < 8) filter_channel = 8;

    const int M = output_channel;
    const int N = output_batch * output_height * output_width;
    const int K = filter_channel * filter_height * filter_width;
    constexpr int TLIE_M_PER_BLOCK = 128;
    constexpr int TLIE_N_PER_BLOCK = 128;
    constexpr int TLIE_K_PER_BLOCK = 8;
    constexpr int TLIE_M_PER_THREAD = 8;
    constexpr int TLIE_N_PER_THREAD = 8;
    const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
    const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
                        CEIL_DIV(N, TLIE_N_PER_BLOCK));

    im2winConvCUDAimplentNCHWHPCkernel< TLIE_M_PER_BLOCK,
                                        TLIE_N_PER_BLOCK,
                                        TLIE_K_PER_BLOCK,
                                        TLIE_M_PER_THREAD,
                                        TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
                            (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
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


template< int TLIE_M_PER_BLOCK,
          int TLIE_N_PER_BLOCK,
          int TLIE_K_PER_BLOCK,
          int TLIE_M_PER_THREAD,
          int TLIE_N_PER_THREAD >
__global__ void im2winConvCUDAimplentNHWCHPCkernel(float *inptr, float *filptr, float *outptr, size_t *d_dims_a, size_t *d_dims_b, size_t *d_dims_c){
    const int THREAD_M_PER_BLOCK = TLIE_M_PER_BLOCK / TLIE_M_PER_THREAD;
    const int THREAD_N_PER_BLOCK = TLIE_N_PER_BLOCK / TLIE_N_PER_THREAD;
    const int THREAD_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;

    const int K = d_dims_b[1] * d_dims_b[2] * d_dims_b[3];

    const int tid = threadIdx.x;
    const int tx = tid / THREAD_N_PER_BLOCK;
    const int ty = tid % THREAD_N_PER_BLOCK;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int input_batch = d_dims_a[0];
    const int input_channel = d_dims_a[1];
    const int input_height = d_dims_a[2];
    const int input_width = d_dims_a[3];

    const int filter_batch = d_dims_b[0];
    const int filter_channel = d_dims_b[1];
    const int filter_height = d_dims_b[2];
    const int filter_width = d_dims_b[3];

    const int output_batch = d_dims_c[0];
    const int output_channel = d_dims_c[1];
    const int output_height = d_dims_c[2];
    const int output_width = d_dims_c[3];

    const int stride_height = (input_height - filter_height) / (output_height - 1);
    const int stride_width = (input_width - filter_width) / (output_width - 1);

    float Asmem[TLIE_K_PER_BLOCK*TLIE_N_PER_BLOCK];
    float Bsmem[TLIE_K_PER_BLOCK*TLIE_M_PER_BLOCK];

    float Areg[TLIE_N_PER_THREAD];
    float Breg[TLIE_M_PER_THREAD];
    float Creg[TLIE_M_PER_THREAD*TLIE_N_PER_THREAD];

    const int THREAD_TILE_A_PER_ROW = TLIE_N_PER_BLOCK;
    const int THREAD_TILE_B_PER_ROW = TLIE_M_PER_BLOCK;

    const int TILE_A_ROW = tid / TLIE_N_PER_BLOCK;
    const int TILE_B_ROW = tid / TLIE_M_PER_BLOCK;

    const int TILE_A_COL = tid % TLIE_N_PER_BLOCK;
    const int TILE_B_COL = tid % TLIE_M_PER_BLOCK;

    const int TILE_A_ROW_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_A_PER_ROW;
    const int TILE_B_ROW_STRIDE = THREAD_PER_BLOCK / THREAD_TILE_B_PER_ROW;

    if(THREAD_PER_BLOCK % THREAD_TILE_A_PER_ROW !=0||THREAD_PER_BLOCK % THREAD_TILE_B_PER_ROW!=0){printf("TILE_THREAD_STRIDE ERROR");return;}

    for (int k_count = 0; k_count < K; k_count += TLIE_K_PER_BLOCK){
        for (int i = 0; i < TLIE_K_PER_BLOCK; i += TILE_B_ROW_STRIDE){
            int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + TILE_B_COL;
            int id_ic = (k_count + TILE_B_ROW + i) / (filter_height*filter_width);
            int res_ic = (k_count + TILE_B_ROW + i) % (filter_height*filter_width);
            int id_fh = res_ic / filter_width;
            int id_fw = res_ic % filter_width;

            Bsmem[(TILE_B_ROW+i)*TLIE_M_PER_BLOCK+TILE_B_COL] = 
            filptr[id_oc*filter_channel*filter_height*filter_width
                  +id_ic*filter_height*filter_width
                  +id_fh
                  +id_fw*filter_height];
        }
        // if(tid==0){
        //     printf("Bsmem :");
        //     for (int i = 0; i < TLIE_K_PER_BLOCK; ++i){
        //         for (int j = 0; j < TLIE_M_PER_BLOCK; ++j){
        //             printf(" %f",Bsmem[i*TLIE_M_PER_BLOCK+j]);
        //         }
        //         printf("\n");
        //     }
        // }
        for (int j = 0; j < TLIE_K_PER_BLOCK; j += TILE_A_ROW_STRIDE){
            int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + TILE_A_COL)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + TILE_A_COL)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            int id_ic = (k_count + TILE_A_ROW + j) / (filter_height*filter_width);
            int res_ic = (k_count + TILE_A_ROW + j) % (filter_height*filter_width);
            int id_fh = res_ic / filter_width;
            int id_fw = res_ic % filter_width;

            int id_ih = id_oh*filter_height*input_width+id_fh;
            int id_iw = id_ow*stride_width+id_fw;

            Asmem[(TILE_A_ROW+j)*TLIE_N_PER_BLOCK+TILE_A_COL] = 
            inptr[id_ob*input_channel*output_height*filter_height*filter_width
                 +id_ic*output_height*filter_height*input_width
                 +id_ih
                 +id_iw*filter_height];
        }
            if((bx==0&&by==0)&&tid==0){
            printf("Asmem : ");
            printf("input_width * filter_height :%d\n",input_width * filter_height);
            for (int i = 0; i < input_width * filter_height; ++i){
                for (int j = 0; j < TLIE_K_PER_BLOCK * TLIE_N_PER_BLOCK; ++j){
                    printf(" %f",Asmem[i*TLIE_K_PER_BLOCK * TLIE_N_PER_BLOCK+j]);
                }
                printf("\n");
            }
        }

        if((bx==0&&by==0)&&tid==0){
            printf("Bsmem :");
            for (int i = 0; i < filter_height * filter_width; ++i){
                for (int j = 0; j < TLIE_K_PER_BLOCK * TLIE_M_PER_BLOCK; ++j){
                    printf(" %f",Bsmem[i*TLIE_K_PER_BLOCK * TLIE_M_PER_BLOCK+j]);
                }
                printf("\n");
            }
        }
        for (int inner_k = 0; inner_k < TLIE_K_PER_BLOCK; ++inner_k){
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                Breg[i] = Bsmem[inner_k*TLIE_M_PER_BLOCK+tx*TLIE_M_PER_THREAD+i];
            }
            for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                Areg[j] = Asmem[inner_k*TLIE_N_PER_BLOCK+ty*TLIE_N_PER_THREAD+j];
            }
            for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
                for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
                    Creg[i*TLIE_N_PER_THREAD+j] +=
                    Breg[i] * Areg[j];                                      
                }
            }
        }
    }
    
    for (int i = 0; i < TLIE_M_PER_THREAD; ++i){
        for (int j = 0; j < TLIE_N_PER_THREAD; ++j){
            int id_oc = bx * TLIE_M_PER_BLOCK + tx * TLIE_M_PER_THREAD + i;
            int id_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)/(output_height * output_width);
            int res_ob = (by * TLIE_N_PER_BLOCK + ty * TLIE_N_PER_THREAD + j)%(output_height * output_width);
            int id_oh = res_ob / output_width;
            int id_ow = res_ob % output_width;

            outptr[id_ob*output_channel*output_height*output_width
            +id_oc*output_height*output_width
            +id_oh*output_width
            +id_ow] = Creg[i*TLIE_N_PER_THREAD+j];
        }
    }
    //printf("Idx_x : %d\n", Idx_x);
    // printf("Idx_y : %d\n", Idx_y);
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
    const int M = dims_c[1]; //1
    const int N = dims_c[0] * dims_c[2] * dims_c[3]; //1 * 2 * 2
    const int K = dims_b[1] * dims_b[2] * dims_b[3]; //1 * 2 * 2
    constexpr int TLIE_M_PER_BLOCK = 4;
    constexpr int TLIE_N_PER_BLOCK = 4;
    constexpr int TLIE_K_PER_BLOCK = 4;
    constexpr int TLIE_M_PER_THREAD = 1;
    constexpr int TLIE_N_PER_THREAD = 1;
    const dim3 blockSize((TLIE_M_PER_BLOCK/TLIE_M_PER_THREAD) * (TLIE_N_PER_BLOCK/TLIE_N_PER_THREAD));
    const dim3 gridSize(CEIL_DIV(M, TLIE_M_PER_BLOCK), 
                        CEIL_DIV(N, TLIE_N_PER_BLOCK));

    im2winConvCUDAimplentNHWCHPCkernel< TLIE_M_PER_BLOCK,
                                        TLIE_N_PER_BLOCK,
                                        TLIE_K_PER_BLOCK,
                                        TLIE_M_PER_THREAD,
                                        TLIE_N_PER_THREAD ><<<gridSize, blockSize>>>
                            (inptr, filptr, outptr, d_dims_a, d_dims_b, d_dims_c);
    //result to host
    //cudaMemcpy(outptr, d_outptr, sizeof(float)*dims_c[0]*dims_c[1]*dims_c[2]*dims_c[3], cudaMemcpyDeviceToHost);
    //clean device memory
    cudaFree(d_dims_a);
    cudaFree(d_dims_b);
    cudaFree(d_dims_c);
}

template<>
int im2winConvCUDAimplentNHWCHPC(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

    return 0;
}

template<>
int im2winConvCUDAimplentNHWCBLASHPC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){

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
    
    int BLOCK = 4;
    int CHUNK = filter_batch / BLOCK;

    int m, n, lda, incx, incy;
    float alpha, beta, *a, *x, *y;
    m = filter_height * filter_width * input_channel;
    n = BLOCK; //BLOCK
    lda = m;
    incx = 1;
    incy = 1;
    alpha = 1;
    beta = 1;

    // #pragma omp parallel for schedule(omp_flag)
    // parallel by stream

    // size_t n_stream = (input_batch - 1) / 256 + 1;
    // cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    // cublasHandle_t *handle = (cublasHandle_t *)malloc(n_stream * sizeof(cublasHandle_t));
    // for(size_t i = 0; i < n_stream; ++i){
    //     cudaStreamCreate(&stream[i]);
    //     cublasCreate(&handle[i]);
    //     cublasSetStream(handle[i], stream[i]);
    // }
    // size_t id_stream = 0;
    size_t handle_num = CHUNK * input_batch * output_height * output_width;
    cublasHandle_t *handle = (cublasHandle_t *)malloc(handle_num * sizeof(cublasHandle_t));
    for(size_t i = 0; i < handle_num; ++i)
        cublasCreate(&handle[i]);

    size_t handle_id = 0;
    nvtxRangePush(__func__);
    //#pragma omp parallel for schedule(dynamic)
    for(int s = 0; s < CHUNK; ++s){
        
    //     //cudaMalloc(&filter_block, sizeof(float) * BLOCK * filter_volume);
    //     cudaMemcpyToSymbol(filter_block, filptr + s * BLOCK * filter_volume, 
    //                        sizeof(float) * BLOCK * filter_volume);
    //     printf("init constant mem\n");
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

                    a = filptr + s * BLOCK * filter_volume;
                    //a = filter_block;
                    x = inptr + jcibD;
                    y = outptr + jibY + s * BLOCK;

                    nvtxRangePush("cublasSgemv call");
                    
                    cublasXgemv(handle[handle_id], CUBLAS_OP_T, m, n, &alpha, a, lda,
                                x, incx, &beta, y, incy);

                    nvtxRangePop();
                    handle_id++;
                    }
                nvtxRangePop();
            }
            nvtxRangePop();
        }
        nvtxRangePop();
     }
     nvtxRangePop();      
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

    size_t n_handle = input_batch * output_height * output_width;
    //cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    //cublasHandle_t *handle = (cublasHandle_t *)malloc(n_handle * sizeof(cublasHandle_t));
    // for(size_t i = 0; i < n_handle; ++i){
    //     //cudaStreamCreate(&stream[i]);
    //     cublasCreate(&handle[i]);
    //     //cublasSetStream(handle[i], stream[i]);
    // }
    size_t id_handle = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    nvtxRangePush(__func__);
    //#pragma omp parallel for schedule(dynamic)
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
                
                cublasXgemv(handle, CUBLAS_OP_T, m, n, &alpha, a, lda,
                            x, incx, &beta, y, incy);
                id_handle++;
                
                nvtxRangePop();
            }
            nvtxRangePop();
        }
        nvtxRangePop();
    }
    nvtxRangePop();      
    return 0;
}

// template<>
// int Deprecate_im2winConvCUDAimplentNHWC(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
//     cublasHandle_t handle;
//     cublasStatus_t status = cublasCreate(&handle);
//     //std::cout<<"im2winNHWC run"<<std::endl;
//     if(status != CUBLAS_STATUS_SUCCESS){
//         if(status == CUBLAS_STATUS_NOT_INITIALIZED){
//             std::cout<<"CUBLAS status error!"<<std::endl;
//         }
//         getchar();
//         return EXIT_FAILURE;
//     }

//     const size_t input_batch = dims_a[0];
//     const size_t input_channel = dims_a[1];
//     const size_t input_height = dims_a[2];
//     const size_t input_width = dims_a[3];

//     const size_t filter_batch = dims_b[0];
//     const size_t filter_channel = dims_b[1];
//     const size_t filter_height = dims_b[2];
//     const size_t filter_width = dims_b[3];

//     const size_t output_batch = dims_c[0];
//     const size_t output_channel = dims_c[1];
//     const size_t output_height = dims_c[2];
//     const size_t output_width = dims_c[3];    

//     const size_t stride_height = (dims_a[2] - dims_b[2]) / (dims_c[2] - 1);
//     const size_t stride_width = (dims_a[3] - dims_b[3]) / (dims_c[3] - 1);

//     const size_t output_csize = output_batch * output_channel * output_height * output_width;
//     const size_t output_area = output_height * output_width;
//     const size_t filter_area = filter_height * filter_width;
//     const size_t filter_volume = filter_area * filter_channel;
//     const size_t gap_width = stride_width * filter_height;
//     const size_t window_row = filter_height * input_width;
//     const size_t window_area = output_height * window_row;
//     const size_t window_volume = input_channel * window_area;
//     const size_t output_volume = filter_batch * output_area;

//     int m, n, lda, incx, incy, batchCount;
//     float alpha, beta, **a, **x, **y;
//     batchCount = input_batch * output_height * output_width;
//     m = filter_height * filter_width * input_channel;
//     n = filter_batch;
//     lda = m;
//     incx = 1;
//     incy = 1;
//     alpha = 1;
//     beta = 1;
//     a = (float **)malloc(sizeof(float*)*batchCount);
//     x = (float **)malloc(sizeof(float*)*batchCount);
//     y = (float **)malloc(sizeof(float*)*batchCount);
//     // #pragma omp parallel for schedule(omp_flag)
//     // parallel by stream
//     // size_t n_stream = input_batch * output_height * output_width;
//     // cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
//     // for(size_t i = 0; i < n_stream; ++i)
//     //     cudaStreamCreate(&stream[i]);
//     // size_t id_stream = 0;

//     //Create pointers of Aarray, Xarray and Yarray;
//     int count = 0;
//     for(int b = 0; b < input_batch; ++b){
//         int bY = b * output_height * output_width * filter_batch;
//         int bD = b * output_height * filter_height * input_width * input_channel;
//         for(int i = 0; i < output_height; ++i){
//             int ibY = bY + i * output_width * filter_batch;
//             int ibD = bD + i * filter_height * input_width * input_channel;
//             for(int j = 0; j < output_width; ++j){
//                 int jibY = ibY + j * filter_batch;
//                 int jcibD = ibD + j * stride_width * filter_height * input_channel;

//                 a[count] = filptr;
//                 x[count] = inptr + jcibD;
//                 y[count] = outptr + jibY;
//                 count++;
//             }
//         }
//     }
//     cublasSgemvBatched(handle, CUBLAS_OP_T, m, n, &alpha, a, lda, x, incx,
//                        &beta, y, incy, batchCount);
//     return 0;
// }

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

template <>
int cuDNNConvNCHW(float *inptr, float *filptr, float *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    // handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    //input
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               dims_a[0], dims_a[1], dims_a[2], dims_a[3]);
    //filter
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(filter_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               dims_b[0], dims_b[1], dims_b[2], dims_b[3]);
    //output
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               dims_c[0], dims_c[1], dims_c[2], dims_c[3]);

    const size_t stride_height = (dims_a[2] - dims_b[2]) / (dims_c[2] - 1);
    const size_t stride_width = (dims_a[3] - dims_b[3]) / (dims_c[3] - 1);

    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    0, 0, // zero-padding
                                    1, 1, // stride
                                    1, 1,
                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    // workspace size && allocate memory
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            filter_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                            &workspace_size);

    void *workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    // convolution
    auto alpha = 1.0f, beta = 1.0f;
    cudnnConvolutionForward(handle,
                            &alpha, input_descriptor, inptr,
                            filter_descriptor, filptr,
                            conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                            workspace, workspace_size,
                            &beta, output_descriptor, outptr);
    // destroy
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);

    cudnnDestroy(handle);

    return 0;
}

template <>
int cuDNNConvNCHW(double *inptr, double *filptr, double *outptr, size_t *dims_a, size_t *dims_b, size_t *dims_c){
    return 0;
}