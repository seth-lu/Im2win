#include "WeTensor.hpp"
#include "Convolution.hpp"
#include "test_benchmark.cpp"

//mkl_enable_instructions(MKL_ENABLE_AVX2)
//  #include "mkl.h"

// int test_s(){
//     STensor tensorA(1,1,3,3), tensorB(1,1,1,1);
//     tensorA.genArrangeTensor();
//     tensorB.genArrangeTensor();

//     auto dataTensor = tensorA.getDataTensor();
//     std::cout<<tensorA.compareTensor(tensorB)<<std::endl;
//     std::cout<<dataTensor<<std::endl;
//     float* dataPtr = tensorA.getDataPtr();
//     float* dataPtr2 = dataTensor.data_ptr<float>();
//     std::cout<<"data_ptr: "<<*dataPtr<<std::endl;

//     WeTensor<float>* A = new STensor(1,1,3,3);
//     WeTensor<float>* B = new STensor(1,1,1,1);
//     WeTensor<float>* C = new STensor(1,1,3,3);
//     WeTensor<float>* C2 = new STensor(1,1,3,3);
//     WeTensor<float>* C3 = new STensor(1,1,3,3);
//     WeTensor<float>* C4 = new STensor(1,1,3,3);

//     A->initDataTensor();
//     B->initDataTensor();
//     C->setZeroTensor();
//     C2->setZeroTensor();
//     C3->setZeroTensor();
//     C4->setZeroTensor();

//     std::cout<<"before_conv: "<<C->getDataTensor()<<std::endl;
//     Convolution<float>* conv = new Im2colConv<float>(A, B, C, 1);
//     conv->conv_implement();
//     std::cout<<"after_conv: "<<C->getDataTensor()<<std::endl;

//     std::cout<<"before_conv2: "<<C2->getDataTensor()<<std::endl;
//     Convolution<float>* conv2 = new DirectConv<float>(A, B, C2, 1);
//     conv2->conv_implement();
//     std::cout<<"after_conv2: "<<C2->getDataTensor()<<std::endl;

//     std::cout<<"before_conv3: "<<C3->getDataTensor()<<std::endl;
//     Convolution<float>* conv3 = new Im2winConvBase<float>(A, B, C3, 1);
//     conv3->conv_implement();
//     std::cout<<"after_conv3: "<<C3->getDataTensor()<<std::endl;

//     std::cout<<"before_conv4: "<<C4->getDataTensor()<<std::endl;
//     Convolution<float>* conv4 = new Im2winConvSIMD<float>(A, B, C4, 1);
//     conv4->conv_implement();
//     std::cout<<"after_conv4: "<<C4->getDataTensor()<<std::endl;
//     conv4->get_gflops();
//     std::cout<<"gflops: "<<conv->get_gflops()<<std::endl;
//     std::cout<<"gflops: "<<conv2->get_gflops()<<std::endl;

//     test_benchmarks<double>();
//     return 0;
// }

// int test_d(){
//     DTensor tensorA(1,1,3,3), tensorB(1,1,1,1);
//     tensorA.genArrangeTensor();
//     tensorB.genArrangeTensor();

//     auto dataTensor = tensorA.getDataTensor();
//     std::cout<<tensorA.compareTensor(tensorB)<<std::endl;
//     std::cout<<dataTensor<<std::endl;
//     double* dataPtr = tensorA.getDataPtr();
//     double* dataPtr2 = dataTensor.data_ptr<double>();
//     std::cout<<"data_ptr: "<<*dataPtr<<std::endl;

//     WeTensor<double>* A = new DTensor(1,1,3,3);
//     WeTensor<double>* B = new DTensor(1,1,1,1);
//     WeTensor<double>* C = new DTensor(1,1,3,3);
//     WeTensor<double>* C2 = new DTensor(1,1,3,3);
//     WeTensor<double>* C3 = new DTensor(1,1,3,3);
//     WeTensor<double>* C4 = new DTensor(1,1,3,3);

//     A->initDataTensor();
//     B->initDataTensor();
//     C->setZeroTensor();
//     C2->setZeroTensor();
//     C3->setZeroTensor();
//     C4->setZeroTensor();

//     std::cout<<"before_conv: "<<C->getDataTensor()<<std::endl;
//     Convolution<double>* conv = new Im2colConv<double>(A, B, C, 1);
//     conv->conv_implement();
//     std::cout<<"after_conv: "<<C->getDataTensor()<<std::endl;

//     std::cout<<"before_conv2: "<<C2->getDataTensor()<<std::endl;
//     Convolution<double>* conv2 = new DirectConv<double>(A, B, C2, 1);
//     conv2->conv_implement();
//     std::cout<<"after_conv2: "<<C2->getDataTensor()<<std::endl;

//     std::cout<<"before_conv3: "<<C3->getDataTensor()<<std::endl;
//     Convolution<double>* conv3 = new Im2winConvBase<double>(A, B, C3, 1);
//     conv3->conv_implement();
//     std::cout<<"after_conv3: "<<C3->getDataTensor()<<std::endl;

//     std::cout<<"before_conv4: "<<C4->getDataTensor()<<std::endl;
//     Convolution<double>* conv4 = new Im2winConvSIMD<double>(A, B, C4, 1);
//     conv4->conv_implement();
//     std::cout<<"after_conv4: "<<C4->getDataTensor()<<std::endl;

//     std::cout<<"gflops: "<<conv3->get_gflops()<<std::endl;
//     std::cout<<"gflops: "<<conv4->get_gflops()<<std::endl;
//     return 0;
// }

int test_filter(){
    // STensor tensorA(1,2,3,3), tensorB(2,2,2,2);
    // tensorA.genArrangeTensor();
    // tensorB.genArrangeTensor();

    // auto dataTensor = tensorA.getDataTensor();
    // std::cout<<tensorA.compareTensor(tensorB)<<std::endl;
    // std::cout<<dataTensor<<std::endl;
    // float* dataPtr = tensorA.getDataPtr();
    // float* dataPtr2 = dataTensor.data_ptr<float>();
    // std::cout<<"data_ptr: "<<*dataPtr<<std::endl;
    size_t i_b, i_c, i_h, i_w, f_b, f_c, f_h, f_w, stride, o_b, o_c, o_h, o_w;
    i_b = 2;
    i_c = 3;
    i_h = 5;
    i_w = i_h;
    f_b = 2;
    f_c = i_c;
    f_h = 3;
    f_w = f_h;
    stride = 2;
    o_b = i_b;
    o_c = f_b;
    o_h = (i_h - f_h)/stride + 1;
    o_w = (i_w - f_w)/stride + 1;

    WeTensor<float>* A = new STensor(i_b,i_c,i_h,i_w);
    WeTensor<float>* B = new STensor(f_b,f_c,f_h,f_w);
    WeTensor<float>* C = new STensor(o_b,o_c,o_h,o_w);

    WeTensor<float>* A2 = new STensor(i_b,i_c,i_h,i_w);
    WeTensor<float>* B2 = new STensor(f_b,f_c,f_h,f_w);
    WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);
    //WeTensor<float>* C3 = new STensor(o_b,o_c,o_h,o_w);
    // WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);
    // WeTensor<float>* C2 = new STensor(1,1,3,3);
    // WeTensor<float>* C3 = new STensor(1,1,3,3);
    // WeTensor<float>* C4 = new STensor(1,1,3,3);

    A->genArrangeTensor();
    B->genArrangeTensor();
    C->setZeroTensor();
    A->deviceToGPU();
    B->deviceToGPU();
    C->deviceToGPU();


    A2->genArrangeTensor();
    B2->genArrangeTensor();
    C2->setZeroTensor();
    //C3->setZeroTensor();

    A2->deviceToGPU();
    B2->deviceToGPU();
    C2->deviceToGPU();

    // A2->channelsLast();
    // B2->channelsLast();
    // C2->channelsLast();
    // C3->channelsLast();
    // C2->setZeroTensor();

    // std::cout<<"before_conv: "<<C->getDataTensor()<<std::endl;

    Convolution<float>* conv_im2col = new Im2colConv<float>(A2, B2, C2, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(A, B, C, stride);
    //Convolution<float>* conv_im2win2 = new Im2winConvHPC<float>(A2, B2, C2, 1);
    // conv->conv_implement();
    // Convolution<float>* conv2 = new DirectConvCUDA<float>(A, B, C2, 1);
    std::cout<<"direct start"<<std::endl;
    torch::cuda::synchronize();
    auto start1 = std::chrono::steady_clock::now();
    conv_direct->conv_implement();
    torch::cuda::synchronize();
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end1 - start1;
    double tmp1 = elapsed_seconds.count();
    std::cout<<"direct end"<<std::endl;
    torch::cuda::synchronize();
    auto start2 = std::chrono::steady_clock::now();
    conv_im2col->conv_implement();
    torch::cuda::synchronize();
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
    double tmp2 = elapsed_seconds2.count();
    std::cout<<"im2col end"<<std::endl;
    std::cout<<C->getDataTensor()<<std::endl;
    std::cout<<C2->getDataTensor()<<std::endl;
    // int iter = o_b*o_c*o_h*o_w;
    // for(int i = 0; i < iter; ++i)
    //     std::cout<<C->getDataPtr()[i]<<" ";
    // std::cout<<"------------------"<<std::endl;
    // for(int i = 0; i < iter; ++i)
    //     std::cout<<C2->getDataPtr()[i]<<" ";
    // std::cout<<"------------------"<<std::endl;
    // for(int i = 0; i < iter; ++i)
    //     std::cout<<C3->getDataPtr()[i]<<" ";
    // std::cout<<"after_conv: "<<C->getDataTensor()<<std::endl;
    // std::cout<<"after_conv: "<<C2->getDataTensor()<<std::endl;
    std::cout<<"time1: "<<tmp1<<std::endl;
    std::cout<<"time2: "<<tmp2<<std::endl;
    // std::cout<<"C1 and C3: "<<C->compareTensor(*C3)<<std::endl;
    // std::cout<<"C2 and C3: "<<C2->compareTensor(*C3)<<std::endl;
    // std::cout<<"after_conv: "<<C2->getDataTensor()<<std::endl;
    // std::cout<<C->compareTensor(*C2)<<std::endl;
    // delete A;
    // A = nullptr;
    delete B;
    delete C;
    // delete C2;
    
    // delete conv;
    // delete conv2;
    return 0;
}

int test_mkl(){
    double *A, *B, *C;
    int m, n, p, i, j;
    double alpha, beta;

    m = 10; p = 10; n = 10;
    alpha = 1.0; beta = 0.0;

    A = (double *)malloc(m*p*sizeof(double));
    B = (double *)malloc(p*n*sizeof(double));
    C = (double *)malloc(m*n*sizeof(double));

    for(i = 0; i < (m*p); i++)
        A[i] = (double)(i+1);
    for(i = 0; i < (p*n); i++)
        B[i] = (double)(i+1);
    for(i = 0; i < (m*n); i++)
        C[i] = (double)(i+1);    
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, A, p, B, n, beta, C, n);
    std::cout<<"matirx A :"<<std::endl;
    for(i = 0; i < m; i++){
        for(j = 0; j < p; j++){
            std::cout<<A[i * p + j]<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<"matirx B :"<<std::endl;
    for(i = 0; i < m; i++){
        for(j = 0; j < p; j++){
            std::cout<<B[i * p + j]<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<"matirx C :"<<std::endl;
    for(i = 0; i < m; i++){
        for(j = 0; j < p; j++){
            std::cout<<C[i * p + j]<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}

int nchw(){
    size_t i_b, i_c, i_h, i_w;
    i_b = 1;
    i_c = 3;
    i_h = 4;
    i_w = i_h;


    WeTensor<float>* A = new STensor(i_b,i_c,i_h,i_w);

    // WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);
    // WeTensor<float>* C2 = new STensor(1,1,3,3);
    // WeTensor<float>* C3 = new STensor(1,1,3,3);
    // WeTensor<float>* C4 = new STensor(1,1,3,3);

    A->genArrangeTensor();
    std::cout<<A->getDataTensor().sizes()<<std::endl;
    std::cout<<A->getDataTensor().strides()<<std::endl;
    auto B = A->getDataTensor().to(at::MemoryFormat::ChannelsLast);
    A->channelsLast();
    std::cout<<B.sizes()<<std::endl;
    std::cout<<B.strides()<<std::endl;
    std::cout<<A->getDataTensor()<<std::endl;
    std::cout<<B<<std::endl;
    for(int i = 0; i < 48; ++i)
        std::cout<<*(A->getDataTensor().data_ptr<float>()+i)<<" ";
    std::cout<<"---------------"<<std::endl;
    for(int i = 0; i < 48; ++i)
        std::cout<<*(B.data_ptr<float>()+i)<<" ";

    // B->genArrangeTensor();
    // C->setZeroTensor();
    // C2->setZeroTensor();

    // std::cout<<"before_conv: "<<C->getDataTensor()<<std::endl;
    // Convolution<float>* conv = new Im2winConvHPC<float>(A, B, C, 1);
    // conv->conv_implement();
    // Convolution<float>* conv2 = new DirectConvCUDA<float>(A, B, C2, 1);
    // conv2->conv_implement();
    // std::cout<<"after_conv: "<<C->getDataTensor()<<std::endl;
    // std::cout<<"after_conv: "<<C2->getDataTensor()<<std::endl;
    // std::cout<<C->compareTensor(*C2)<<std::endl;
    delete A;
    A = nullptr;

    // delete C2;
    
    // delete conv;
    // delete conv2;
    return 0;
}

int nhwc(){
size_t i_b, i_c, i_h, i_w, f_b, f_c, f_h, f_w, stride, o_b, o_c, o_h, o_w;
    i_b = 6;
    i_c = 6;
    i_h = 7;
    i_w = i_h;
    f_b = 6;
    f_c = i_c;
    f_h = 3;
    f_w = f_h;
    stride = 1;
    o_b = i_b;
    o_c = f_b;
    o_h = (i_h - f_h)/stride + 1;
    o_w = (i_w - f_w)/stride + 1;

    WeTensor<float>* A = new STensor(i_b,i_c,i_h,i_w);
    WeTensor<float>* B = new STensor(f_b,f_c,f_h,f_w);
    WeTensor<float>* C = new STensor(o_b,o_c,o_h,o_w);
    WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);

    // WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);
    // WeTensor<float>* C2 = new STensor(1,1,3,3);
    // WeTensor<float>* C3 = new STensor(1,1,3,3);
    // WeTensor<float>* C4 = new STensor(1,1,3,3);

    A->genArrangeTensor();
    B->genArrangeTensor();
    C->setZeroTensor();
    C2->setZeroTensor();


    A->channelsLast();
    B->channelsLast();
    C->channelsLast();
    C2->channelsLast();
    
    // C2->setZeroTensor();

    // std::cout<<"before_conv: "<<C->getDataTensor()<<std::endl;

    Convolution<float>* conv_im2col = new Im2colConv<float>(A, B, C, 1);
    Convolution<float>* conv_direct = new DirectConv<float>(A, B, C2, 1);

    // conv->conv_implement();
    // Convolution<float>* conv2 = new DirectConvCUDA<float>(A, B, C2, 1);
    auto start1 = std::chrono::steady_clock::now();
    conv_im2col->conv_implement();
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end1 - start1;
    double tmp1 = elapsed_seconds.count();

    auto start2 = std::chrono::steady_clock::now();
    conv_direct->conv_implement();
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
    double tmp2 = elapsed_seconds2.count();

    std::cout<<"max differ: "<<C->compareTensor(*C2)<<std::endl;

    delete A;
    return 0;
}

int test_cuda(){
    size_t i_b, i_c, i_h, i_w, f_b, f_c, f_h, f_w, stride, o_b, o_c, o_h, o_w;
    i_b = 16;
    i_c = 1;
    i_h = 4;
    i_w = i_h;
    f_b = 16;
    f_c = i_c;
    f_h = 3;
    f_w = f_h;
    stride = 1;
    o_b = i_b;
    o_c = f_b;
    o_h = (i_h - f_h)/stride + 1;
    o_w = (i_w - f_w)/stride + 1;

    WeTensor<float>* A = new STensor(i_b,i_c,i_h,i_w);
    WeTensor<float>* B = new STensor(f_b,f_c,f_h,f_w);
    WeTensor<float>* C = new STensor(o_b,o_c,o_h,o_w);
    WeTensor<float>* C2 = new STensor(o_b,o_c,o_h,o_w);
    WeTensor<float>* C3 = new STensor(o_b,o_c,o_h,o_w);

    A->initDataTensor();
    B->initDataTensor();
    // A->genArrangeTensor();
    // B->genArrangeTensor();
    C->setZeroTensor();
    C2->setZeroTensor();
    C3->setZeroTensor();
    
    // A->channelsLast();
    // B->channelsLast();
    // C->channelsLast();
    // C2->channelsLast();
    // C3->channelsLast();
    // std::cout<<A->getDataTensor()<<std::endl;
    A->deviceToGPU();
    B->deviceToGPU();
    C->deviceToGPU();
    C2->deviceToGPU();
    C3->deviceToGPU();
    // std::cout<<A->getDataTensor()<<std::endl;
    
    std::cout<<"init conv_im2win start!"<<std::endl;
    Convolution<float>* conv_im2col = new Im2colConv<float>(A, B, C, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(A, B, C2, stride);
    Convolution<float>* conv_im2win = new Im2winConvHPC<float>(A, B, C3, stride);
    
    std::cout<<"init conv_im2win complete!"<<std::endl;
    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2win->conv_implement();
    torch::cuda::synchronize();
    
    double max_diff_im2winHPC, max_diff_direct, max_diff_im2winConvSIMD, max_diff_im2winConvHPC;
    max_diff_direct = C->compareTensor(*C2);
    max_diff_im2winHPC = C->compareTensor(*C3);
    std::cout<<"max_diff_direct: "<<max_diff_direct<<std::endl;
    std::cout<<"max_diff_im2winHPC: "<<max_diff_im2winHPC<<std::endl;
    //auto C3 = at::conv2d(A->getDataTensor(), B->getDataTensor(), {}, stride);
    std::cout<<C->getDataTensor()<<std::endl;
    //std::cout<<C2->getDataTensor()<<std::endl;
    std::cout<<C3->getDataTensor()<<std::endl;
    size_t iter = 1;
    auto input = A;
    auto filter = B;
    auto output_im2col = C;
    auto output_direct = C2;
    auto output_im2win = C3;
    double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2win = minIm2winHPCConv(iter, input, filter, output_im2win, stride);
    std::cout<<"run timeim2col: "<<timeIm2col<<std::endl;
    std::cout<<"run timediret: "<<timeDirect<<std::endl;
    std::cout<<"run timeim2win: "<<timeIm2win<<std::endl;

    return 0;
}

int main(){
    std::cout<<torch::show_config()<<std::endl;
    //std::cout<<"CUDA::is_available(): "<<torch::cuda::is_available()<<std::endl;
    test_cuda();
    //test_benchmarks<float>();
    //test_mkl();
    //test_filter();
    //test_conv8<float>(128);
    //nchw();
    //nhwc();
    //testStream();
    return 0;
}