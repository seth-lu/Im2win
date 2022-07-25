#ifndef _TEST_BENCHMARK_
#define _TEST_BENCHMARK_
#include "WeTensor.hpp"
#include "Convolution.hpp"
#include <typeinfo>

template<class dataType>
double minIm2colConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    static double min = -1;
    Convolution<dataType>* im2colConv = new Im2colConv<dataType>(input, filter, output, stride);
    for(size_t i = 0; i < iter; i++){
        auto start = std::chrono::steady_clock::now();
        im2colConv->conv_implement();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        tmp = elapsed_seconds.count();
        if(min > 0){
            min = min < tmp ? min : tmp;
        }
        else min = tmp;
    }
    return min;
}

template<class dataType>
double minDirectConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    static double min = -1;
    Convolution<dataType>* directConv = new DirectConv<dataType>(input, filter, output, stride);
    for(size_t i = 0; i < iter; i++){
        auto start = std::chrono::steady_clock::now();
        directConv->conv_implement();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        tmp = elapsed_seconds.count();
        if(min > 0){
            min = min < tmp ? min : tmp;
        }
        else min = tmp;
    }
    return min;
}

template<class dataType>
double minIm2winBaseConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    static double min = -1;
    Convolution<dataType>* im2winConvBase = new Im2winConvBase<dataType>(input, filter, output, stride);
    for(size_t i = 0; i < iter; i++){
        auto start = std::chrono::steady_clock::now();
        im2winConvBase->conv_implement();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        tmp = elapsed_seconds.count();
        if(min > 0){
            min = min < tmp ? min : tmp;
        }
        else min = tmp;
    }
    return min;
}

template<class dataType>
double minIm2winSIMDConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    static double min = -1;
    Convolution<dataType>* im2winConvSIMD = new Im2winConvSIMD<dataType>(input, filter, output, stride);
    for(size_t i = 0; i < iter; i++){
        auto start = std::chrono::steady_clock::now();
        im2winConvSIMD->conv_implement();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        tmp = elapsed_seconds.count();
        if(min > 0){
            min = min < tmp ? min : tmp;
        }
        else min = tmp;
    }
    return min;
}

template<class dataType>
int test_enter();

// template<class dataType>
// int test_implement(size_t iter, WeTensor<dataType> *input,WeTensor<dataType> *filter,WeTensor<dataType> *output, size_t *dims_a, size_t *dims_b, size_t stride);
int test_implement(size_t iter, WeTensor<float> *input,WeTensor<float> *filter,WeTensor<float> *output, size_t *dims_a, size_t *dims_b, size_t stride){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    input = new STensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    filter = new STensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();

    WeTensor<float>* output_im2col = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_direct = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2win = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2winSIMD = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_direct->setZeroTensor();
    output_im2win->setZeroTensor();
    output_im2winSIMD->setZeroTensor();

    Convolution<float>* conv_im2col = new Im2colConv<float>(input, filter, output_im2col, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(input, filter, output_direct, stride);
    Convolution<float>* conv_im2win = new Im2winConvBase<float>(input, filter, output_im2win, stride);
    Convolution<float>* conv_im2winSIMD = new Im2winConvSIMD<float>(input, filter, output_im2winSIMD, stride);

    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2win->conv_implement();
    conv_im2winSIMD->conv_implement();

    double max_diff_im2winConv, max_diff_direct, max_diff_im2winConvSIMD;
    max_diff_direct = output_im2col->compareTensor(*output_direct);
    max_diff_im2winConv = output_im2col->compareTensor(*output_im2win);
    max_diff_im2winConvSIMD = output_im2col->compareTensor(*output_im2winSIMD);
    
    std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConv : "<<max_diff_im2winConv<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvSIMD : "<<max_diff_im2winConvSIMD<<std::endl;
    std::cout<<"               "<<std::endl;

    double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2winBase = minIm2winBaseConv(iter, input, filter, output_im2win, stride);
    double timeIm2winSIMD = minIm2winSIMDConv(iter, input, filter, output_im2winSIMD, stride);
    
    auto gflops = conv_im2col->get_gflops();
   
    double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    double GFLOPS_Im2winBase = gflops/timeIm2winBase;
    double GFLOPS_Im2winSIMD = gflops/timeIm2winSIMD;

    std::cout<<"run_time_im2col is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    std::cout<<"run_time_im2win is :"<<timeIm2winBase<<"s"<<std::endl;
    std::cout<<"run_time_im2winSIMD is :"<<timeIm2winSIMD<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    std::cout<<"GFLOPS_Im2winBase is :"<<GFLOPS_Im2winBase<<std::endl;
    std::cout<<"GFLOPS_Im2winSIMD is :"<<GFLOPS_Im2winSIMD<<std::endl;

    return 0;
}

int test_implement(size_t iter, WeTensor<double> *input,WeTensor<double> *filter,WeTensor<double> *output, size_t *dims_a, size_t *dims_b, size_t stride){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    input = new DTensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    filter = new DTensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();

    WeTensor<double>* output_im2col = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_direct = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_im2win = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_im2winSIMD = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_direct->setZeroTensor();
    output_im2win->setZeroTensor();
    output_im2winSIMD->setZeroTensor();

    Convolution<double>* conv_im2col = new Im2colConv<double>(input, filter, output_im2col, stride);
    Convolution<double>* conv_direct = new DirectConv<double>(input, filter, output_direct, stride);
    Convolution<double>* conv_im2win = new Im2winConvBase<double>(input, filter, output_im2win, stride);
    Convolution<double>* conv_im2winSIMD = new Im2winConvSIMD<double>(input, filter, output_im2winSIMD, stride);
    
    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2win->conv_implement();
    conv_im2winSIMD->conv_implement();

    double max_diff_im2winConv, max_diff_direct, max_diff_im2winConvSIMD;
    max_diff_direct = output_im2col->compareTensor(*output_direct);
    max_diff_im2winConv = output_im2col->compareTensor(*output_im2win);
    max_diff_im2winConvSIMD = output_im2col->compareTensor(*output_im2winSIMD);
    
    std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConv : "<<max_diff_im2winConv<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvSIMD : "<<max_diff_im2winConvSIMD<<std::endl;
    std::cout<<"               "<<std::endl;

    double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2winBase = minIm2winBaseConv(iter, input, filter, output_im2win, stride);
    double timeIm2winSIMD = minIm2winSIMDConv(iter, input, filter, output_im2winSIMD, stride);
    
    auto gflops = conv_im2col->get_gflops();

    double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    double GFLOPS_Im2winBase = gflops/timeIm2winBase;
    double GFLOPS_Im2winSIMD = gflops/timeIm2winSIMD;

    std::cout<<"run_time_im2col is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    std::cout<<"run_time_im2win is :"<<timeIm2winBase<<"s"<<std::endl;
    std::cout<<"run_time_im2winSIMD is :"<<timeIm2winSIMD<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    std::cout<<"GFLOPS_Im2winBase is :"<<GFLOPS_Im2winBase<<std::endl;
    std::cout<<"GFLOPS_Im2winSIMD is :"<<GFLOPS_Im2winSIMD<<std::endl;

    return 0;
}

template<class dataType>
int test_enter(size_t *dims_a, size_t *dims_b, size_t stride){
    size_t iter = 5;
    WeTensor<dataType> *input, *filter, *output;
    test_implement(iter, input, filter, output, dims_a, dims_b, stride);
}

template<class dataType>
int test_conv1(){
    size_t dims_a[4] = {128, 3, 227, 227};
    size_t dims_b[4] = {96, 3, 11, 11};
    size_t stride = 4;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv1-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv2(){
    size_t dims_a[4] = {128, 3, 231, 231};
    size_t dims_b[4] = {96, 3, 11, 11};
    size_t stride = 4;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv2-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv3(){
    size_t dims_a[4] = {128, 3, 227, 227};
    size_t dims_b[4] = {64, 3, 7, 7};
    size_t stride = 2;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv3-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv4(){
    size_t dims_a[4] = {128, 64, 224, 224};
    size_t dims_b[4] = {64, 64, 7, 7};
    size_t stride = 2;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv4-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv5(){
    size_t dims_a[4] = {128, 96, 24, 24};
    size_t dims_b[4] = {256, 96, 5, 5};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv5-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv6(){
    size_t dims_a[4] = {128, 256, 12, 12};
    size_t dims_b[4] = {512, 256, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv6-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv7(){
    size_t dims_a[4] = {128, 3, 224, 224};
    size_t dims_b[4] = {64, 3, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv7-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv8(){
    size_t dims_a[4] = {128, 64, 112, 112};
    size_t dims_b[4] = {128, 64, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv8-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv9(){
    size_t dims_a[4] = {128, 64, 56, 56};
    size_t dims_b[4] = {64, 64, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv9-----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv10(){
    size_t dims_a[4] = {128, 128, 28, 28};
    size_t dims_b[4] = {128, 128, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv10----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv11(){
    size_t dims_a[4] = {128, 256, 14, 14};
    size_t dims_b[4] = {256, 256, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv11----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_conv12(){
    size_t dims_a[4] = {128, 512, 7, 7};
    size_t dims_b[4] = {512, 512, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv12----"<<std::endl;
    test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return 0;
}

template<class dataType>
int test_benchmarks(){
    test_conv1<dataType>();
    test_conv2<dataType>();
    test_conv3<dataType>();
    test_conv4<dataType>();
    test_conv5<dataType>();
    test_conv6<dataType>();
    test_conv7<dataType>();
    test_conv8<dataType>();
    test_conv9<dataType>();
    test_conv10<dataType>();
    test_conv11<dataType>();
    test_conv12<dataType>();
    return 0;
}

#endif