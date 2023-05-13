#ifndef _TEST_BENCHMARK_
#define _TEST_BENCHMARK_
#include "../src/cpp/WeTensor.hpp"
#include "../src/cpp/Convolution.hpp"
#include <typeinfo>
#include <fstream>
#include <time.h>
#include <string>
#include <vector>
#include <typeinfo>

#define CLEANUP()                 \
    do{                           \
        delete input;             \
        delete filter;            \
        delete output_im2col;     \
        delete output_direct;     \
        delete output_im2winBASE; \
        delete output_im2winHPC; \
        delete conv_im2col;       \
        delete conv_direct;       \
        delete conv_im2winBASE;   \
        delete conv_im2winHPC;   \
        } while (0)                                              
                           

using namespace std;
template<class dataType>
double minIm2colConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    double min = -1;
    Convolution<dataType>* im2colConv = new Im2colConv<dataType>(input, filter, output, stride);
    if(input->deviceStorage == cpu){
        timer t1;
        for(size_t i = 0; i < iter; i++){
            //auto start = std::chrono::steady_clock::now();
            t1.start();
            im2colConv->conv_implement();
            auto end = std::chrono::steady_clock::now();
            //std::chrono::duration<double> elapsed_seconds = end - start;
            t1.end();
            //tmp = elapsed_seconds.count();
            tmp = t1.gettime();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }

    else if(input->deviceStorage == gpu){
        timer t1;
        for(size_t i = 0; i < iter; i++){
            warmUp();
            torch::cuda::synchronize();
            //auto start = std::chrono::steady_clock::now();
            t1.start();
            im2colConv->conv_implement();
            torch::cuda::synchronize();
            //auto end = std::chrono::steady_clock::now();
            t1.end();
            //std::chrono::duration<double> elapsed_seconds = end - start;
            //tmp = elapsed_seconds.count();
            tmp = t1.gettime();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }
    delete im2colConv;
    im2colConv = nullptr;
    return min;
}

template<class dataType>
double averIm2colConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;

    Convolution<dataType>* im2colConv = new Im2colConv<dataType>(input, filter, output, stride);
    if(input->deviceStorage == cpu){
        timer t1;
        t1.start();
        for(size_t i = 0; i < iter; i++){
            im2colConv->conv_implement();        
        }
        t1.end();
        tmp = t1.gettime();
    }

    else if(input->deviceStorage == gpu){
        timer t1;
        torch::cuda::synchronize();
        t1.start();
        for(size_t i = 0; i < iter; i++){
            im2colConv->conv_implement();
        }
        torch::cuda::synchronize();
        t1.end();
        tmp = t1.gettime();
    }
    delete im2colConv;
    im2colConv = nullptr;
    return tmp/iter;
}

template<class dataType>
double minDirectConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    double min = -1;
    Convolution<dataType>* directConv = new DirectConv<dataType>(input, filter, output, stride);
    if(input->deviceStorage == cpu){
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
    }

    else if(input->deviceStorage == gpu){
        for(size_t i = 0; i < iter; i++){
            warmUp();
            torch::cuda::synchronize();
            auto start = std::chrono::steady_clock::now();
            directConv->conv_implement();
            torch::cuda::synchronize();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            tmp = elapsed_seconds.count();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }
    delete directConv;
    directConv = nullptr;
    return min;
}

template<class dataType>
double minIm2winBASEConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    double min = -1;
    Convolution<dataType>* im2winConvBASE = new Im2winConvBASE<dataType>(input, filter, output, stride);
    
    if(input->deviceStorage == cpu){
        for(size_t i = 0; i < iter; i++){
            auto start = std::chrono::steady_clock::now();
            im2winConvBASE->conv_implement();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            tmp = elapsed_seconds.count();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }

    else if(input->deviceStorage == gpu){
        for(size_t i = 0; i < iter; i++){
            warmUp();
            torch::cuda::synchronize();
            auto start = std::chrono::steady_clock::now();
            im2winConvBASE->conv_implement();
            torch::cuda::synchronize();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            tmp = elapsed_seconds.count();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }
    delete im2winConvBASE;
    im2winConvBASE = nullptr;
    return min;
}

template<class dataType>
double minIm2winHPCConv(size_t iter, WeTensor<dataType> *input, WeTensor<dataType> *filter, WeTensor<dataType> *output, size_t stride){
    double tmp;
    double min = -1;
    Convolution<dataType>* im2winConvHPC = new Im2winConvHPC<dataType>(input, filter, output, stride);
    
    if(input->deviceStorage == cpu){
        for(size_t i = 0; i < iter; i++){
            auto start = std::chrono::steady_clock::now();
            im2winConvHPC->conv_implement();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            tmp = elapsed_seconds.count();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }

    else if(input->deviceStorage == gpu){
        for(size_t i = 0; i < iter; i++){
            warmUp();
            torch::cuda::synchronize();
            auto start = std::chrono::steady_clock::now();
            im2winConvHPC->conv_implement();
            torch::cuda::synchronize();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            tmp = elapsed_seconds.count();
            if(min > 0){
                min = min < tmp ? min : tmp;
            }
            else min = tmp;
        }
    }
    delete im2winConvHPC;
    im2winConvHPC = nullptr;
    return min;
}

template<class dataType>
vector<double> test_enter();

//test_directConv_CUDA
vector<double> _test_implement(size_t iter, size_t *dims_a, size_t *dims_b, size_t stride, float flag){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    WeTensor<float>* input = new STensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    WeTensor<float>* filter = new STensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();
    input->deviceToGPU();
    filter->deviceToGPU();
    //test NHWC
    //input->channelsLast();
    //filter->channelsLast();

    WeTensor<float>* output_im2col = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_direct = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2winBASE = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2winHPC = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_im2col->deviceToGPU();
    //output_im2col->channelsLast();

    output_direct->setZeroTensor();
    output_direct->deviceToGPU();
    //output_direct->channelsLast();

    output_im2winBASE->setZeroTensor();
    output_im2winBASE->deviceToGPU();
    //output_im2winBASE->channelsLast();

    output_im2winHPC->setZeroTensor();
    output_im2winHPC->deviceToGPU();
    //output_im2winHPC->channelsLast();

    Convolution<float>* conv_im2col = new Im2colConv<float>(input, filter, output_im2col, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(input, filter, output_direct, stride);
    Convolution<float>* conv_im2winBASE = new Im2winConvBASE<float>(input, filter, output_im2winBASE, stride);
    Convolution<float>* conv_im2winHPC = new Im2winConvHPC<float>(input, filter, output_im2winHPC, stride);

    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2winBASE->conv_implement();
    conv_im2winHPC->conv_implement();
    std::cout<<"conv complete!"<<std::endl;
    // conv_im2win->conv_implement();
    // conv_im2winSIMD->conv_implement();

    double max_diff_im2winConvBASE, max_diff_direct, max_diff_im2winConvSIMD, max_diff_im2winConvHPC;
    max_diff_direct = output_im2col->compareTensor(*output_direct);
    max_diff_im2winConvBASE = output_im2col->compareTensor(*output_im2winBASE);
    max_diff_im2winConvHPC = output_im2col->compareTensor(*output_im2winHPC);
    // //max_diff_im2winConvSIMD = output_im2col->compareTensor(*output_im2winSIMD);
    
    std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvBASE : "<<max_diff_im2winConvBASE<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvHPC : "<<max_diff_im2winConvHPC<<std::endl;
    std::cout<<"               "<<std::endl;
    // //std::cout<<"max_diff_im2winConvSIMD : "<<max_diff_im2winConvSIMD<<std::endl;
    //std::cout<<"               "<<std::endl;

    double timeIm2col = averIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2winBASE = minIm2winBASEConv(iter, input, filter, output_im2winBASE, stride);
    double timeIm2winHPC = minIm2winHPCConv(iter, input, filter, output_im2winHPC, stride);
    // //double timeIm2winSIMD = minIm2winSIMDConv(iter, input, filter, output_im2winSIMD, stride);

    auto gflops = conv_im2col->get_gflops();
   
    double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    double GFLOPS_Im2winBASE = gflops/timeIm2winBASE;
    double GFLOPS_Im2winHPC = gflops/timeIm2winHPC;
    // //double GFLOPS_Im2winSIMD = gflops/timeIm2winSIMD;

    vector<double> log_output;
    std::cout<<"run_time_im2col_CUDA is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    std::cout<<"run_time_im2winBASE is :"<<timeIm2winBASE<<"s"<<std::endl;
    std::cout<<"run_time_im2winHPC is :"<<timeIm2winHPC<<"s"<<std::endl;
    // //std::cout<<"run_time_im2winSIMD is :"<<timeIm2winSIMD<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    std::cout<<"GFLOPS_Im2winBASE is :"<<GFLOPS_Im2winBASE<<std::endl;
    std::cout<<"GFLOPS_Im2winHPC is :"<<GFLOPS_Im2winHPC<<std::endl;
    // //std::cout<<"GFLOPS_Im2winSIMD is :"<<GFLOPS_Im2winSIMD<<std::endl;

    log_output.push_back(timeIm2col);
    log_output.push_back(timeDirect);
    log_output.push_back(timeIm2winBASE);
    log_output.push_back(timeIm2winHPC);

    log_output.push_back(GFLOPS_Im2col);
    log_output.push_back(GFLOPS_Direct);
    log_output.push_back(GFLOPS_Im2winBASE);
    log_output.push_back(GFLOPS_Im2winHPC);

    CLEANUP();
    //c10::cuda::CUDACachingAllocator::empty_cache();
    at::cuda::getPinnedMemoryAllocator();
    return log_output;
}

vector<double> test_implement(size_t iter, size_t *dims_a, size_t *dims_b, size_t stride, float flag){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    WeTensor<float>* input = new STensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    WeTensor<float>* filter = new STensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();
    input->deviceToGPU();
    filter->deviceToGPU();
    //test NHWC
    //input->channelsLast();
    //filter->channelsLast();

    WeTensor<float>* output_im2col = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_direct = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    //WeTensor<float>* output_im2winBASE = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    //WeTensor<float>* output_im2winHPC = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_im2col->deviceToGPU();
    //output_im2col->channelsLast();

    output_direct->setZeroTensor();
    output_direct->deviceToGPU();
    //output_direct->channelsLast();

    //output_im2winBASE->setZeroTensor();
    //output_im2winBASE->deviceToGPU();
    //output_im2winBASE->channelsLast();

    // output_im2winHPC->setZeroTensor();
    // output_im2winHPC->deviceToGPU();
    //output_im2winHPC->channelsLast();

    Convolution<float>* conv_im2col = new Im2colConv<float>(input, filter, output_im2col, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(input, filter, output_direct, stride);
    //Convolution<float>* conv_im2winBASE = new Im2winConvBASE<float>(input, filter, output_im2winBASE, stride);
    //Convolution<float>* conv_im2winHPC = new Im2winConvHPC<float>(input, filter, output_im2winHPC, stride);

    //conv_im2col->conv_implement();
    conv_direct->conv_implement();
    //conv_im2winBASE->conv_implement();
    //conv_im2winHPC->conv_implement();
    std::cout<<"conv complete!"<<std::endl;
    // conv_im2win->conv_implement();
    // conv_im2winSIMD->conv_implement();

    double max_diff_im2winConvBASE, max_diff_direct, max_diff_im2winConvSIMD, max_diff_im2winConvHPC;
    //max_diff_direct = output_im2col->compareTensor(*output_direct);
    //max_diff_im2winConvBASE = output_im2col->compareTensor(*output_im2winBASE);
    //max_diff_im2winConvHPC = output_im2col->compareTensor(*output_im2winHPC);
    // //max_diff_im2winConvSIMD = output_im2col->compareTensor(*output_im2winSIMD);
    
    // std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    // std::cout<<"               "<<std::endl;
    // std::cout<<"max_diff_im2winConvBASE : "<<max_diff_im2winConvBASE<<std::endl;
    // std::cout<<"               "<<std::endl;
    // std::cout<<"max_diff_im2winConvHPC : "<<max_diff_im2winConvHPC<<std::endl;
    // std::cout<<"               "<<std::endl;
    // //std::cout<<"max_diff_im2winConvSIMD : "<<max_diff_im2winConvSIMD<<std::endl;
    //std::cout<<"               "<<std::endl;

    //double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    //double timeIm2winBASE = minIm2winBASEConv(iter, input, filter, output_im2winBASE, stride);
    //double timeIm2winHPC = minIm2winHPCConv(iter, input, filter, output_im2winHPC, stride);
    // //double timeIm2winSIMD = minIm2winSIMDConv(iter, input, filter, output_im2winSIMD, stride);

    auto gflops = conv_im2col->get_gflops();
   
    //double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    //double GFLOPS_Im2winBASE = gflops/timeIm2winBASE;
    //double GFLOPS_Im2winHPC = gflops/timeIm2winHPC;
    // //double GFLOPS_Im2winSIMD = gflops/timeIm2winSIMD;

    vector<double> log_output;
    //std::cout<<"run_time_im2col_CUDA is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    // std::cout<<"run_time_im2winBASE is :"<<timeIm2winBASE<<"s"<<std::endl;
    //std::cout<<"run_time_im2winHPC is :"<<timeIm2winHPC<<"s"<<std::endl;
    // //std::cout<<"run_time_im2winSIMD is :"<<timeIm2winSIMD<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    //std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    // std::cout<<"GFLOPS_Im2winBASE is :"<<GFLOPS_Im2winBASE<<std::endl;
    //std::cout<<"GFLOPS_Im2winHPC is :"<<GFLOPS_Im2winHPC<<std::endl;
    // //std::cout<<"GFLOPS_Im2winSIMD is :"<<GFLOPS_Im2winSIMD<<std::endl;

    //log_output.push_back(timeIm2col);
    // log_output.push_back(timeDirect);
    // log_output.push_back(timeIm2winBASE);
    // log_output.push_back(timeIm2winHPC);

    //log_output.push_back(GFLOPS_Im2col);
    // log_output.push_back(GFLOPS_Direct);
    // log_output.push_back(GFLOPS_Im2winBASE);
    // log_output.push_back(GFLOPS_Im2winHPC);

    //CLEANUP();
    //c10::cuda::CUDACachingAllocator::empty_cache();
    at::cuda::getPinnedMemoryAllocator();
    return log_output;
}

//NHWC
vector<double> test_implement_(size_t iter, size_t *dims_a, size_t *dims_b, size_t stride, float flag){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    WeTensor<float>* input = new STensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    WeTensor<float>* filter = new STensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();

    input->channelsLast();
    filter->channelsLast();

    WeTensor<float>* output_im2col = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_direct = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2winBASE = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<float>* output_im2winHPC = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    //WeTensor<float>* output_im2winSIMD = new STensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_direct->setZeroTensor();
    output_im2winBASE->setZeroTensor();
    output_im2winHPC->setZeroTensor();
    //output_im2winSIMD->setZeroTensor();

    output_im2col->channelsLast();
    output_direct->channelsLast();
    output_im2winBASE->channelsLast();
    output_im2winHPC->channelsLast();
    //output_im2winSIMD->channelsLast();

    Convolution<float>* conv_im2col = new Im2colConv<float>(input, filter, output_im2col, stride);
    Convolution<float>* conv_direct = new DirectConv<float>(input, filter, output_direct, stride);
    Convolution<float>* conv_im2winBASE = new Im2winConvBASE<float>(input, filter, output_im2winBASE, stride);
    Convolution<float>* conv_im2winHPC = new Im2winConvHPC<float>(input, filter, output_im2winHPC, stride);
    //Convolution<float>* conv_im2winSIMD = new Im2winConvSIMD<float>(input, filter, output_im2winSIMD, stride);

    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2winBASE->conv_implement();
    conv_im2winHPC->conv_implement();
    //conv_im2winSIMD->conv_implement();

    double max_diff_im2winConvBASE, max_diff_direct, max_diff_im2winConvSIMD, max_diff_im2winConvHPC;
    max_diff_direct = output_im2col->compareTensor(*output_direct);
    max_diff_im2winConvBASE = output_im2col->compareTensor(*output_im2winBASE);
    max_diff_im2winConvHPC = output_im2col->compareTensor(*output_im2winHPC);
    //max_diff_im2winConvSIMD = output_im2col->compareTensor(*output_im2winSIMD);
    
    std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvBASE : "<<max_diff_im2winConvBASE<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvHPC : "<<max_diff_im2winConvHPC<<std::endl;
    std::cout<<"               "<<std::endl;
    //std::cout<<"max_diff_im2winConvSIMD : "<<max_diff_im2winConvSIMD<<std::endl;
    //std::cout<<"               "<<std::endl;

    double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2winBASE = minIm2winBASEConv(iter, input, filter, output_im2winBASE, stride);
    double timeIm2winHPC = minIm2winHPCConv(iter, input, filter, output_im2winHPC, stride);
    //double timeIm2winSIMD = minIm2winSIMDConv(iter, input, filter, output_im2winSIMD, stride);
    
    auto gflops = conv_im2col->get_gflops();
   
    double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    double GFLOPS_Im2winBASE = gflops/timeIm2winBASE;
    double GFLOPS_Im2winHPC = gflops/timeIm2winHPC;
    //double GFLOPS_Im2winSIMD = gflops/timeIm2winSIMD;

    vector<double> log_output;
    std::cout<<"run_time_im2col is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    std::cout<<"run_time_im2winBASE is :"<<timeIm2winBASE<<"s"<<std::endl;
    std::cout<<"run_time_im2winHPC is :"<<timeIm2winHPC<<"s"<<std::endl;
    //std::cout<<"run_time_im2winSIMD is :"<<timeIm2winSIMD<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    std::cout<<"GFLOPS_Im2winBASE is :"<<GFLOPS_Im2winBASE<<std::endl;
    std::cout<<"GFLOPS_Im2winHPC is :"<<GFLOPS_Im2winHPC<<std::endl;
    //std::cout<<"GFLOPS_Im2winSIMD is :"<<GFLOPS_Im2winSIMD<<std::endl;

    log_output.push_back(timeIm2col);
    log_output.push_back(timeDirect);
    log_output.push_back(timeIm2winBASE);
    log_output.push_back(timeIm2winHPC);

    log_output.push_back(GFLOPS_Im2col);
    log_output.push_back(GFLOPS_Direct);
    log_output.push_back(GFLOPS_Im2winBASE);
    log_output.push_back(GFLOPS_Im2winHPC);

    CLEANUP();
    // delete input;
    // input = NULL;                         
    // delete filter;                       
    // delete output_im2col;         
    // delete output_direct;         
    // delete output_im2winBASE; 
    // delete output_im2winHPC; 
    // delete conv_im2col;             
    // delete conv_direct;             
    // delete conv_im2winBASE;     
    // delete conv_im2winHPC;     

    return log_output;
}

vector<double> test_implement(size_t iter, size_t *dims_a, size_t *dims_b, size_t stride, double flag){
    size_t dims_c[4]={dims_a[0], dims_b[0], (dims_a[2]-dims_b[2])/stride+1, (dims_a[3]-dims_b[3])/stride+1};
    WeTensor<double>* input = new DTensor(dims_a[0],dims_a[1],dims_a[2],dims_a[3]);
    WeTensor<double>* filter = new DTensor(dims_b[0],dims_b[1],dims_b[2],dims_b[3]);

    input->initDataTensor();
    filter->initDataTensor();

    WeTensor<double>* output_im2col = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_direct = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_im2winBASE = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);
    WeTensor<double>* output_im2winHPC = new DTensor(dims_c[0],dims_c[1],dims_c[2],dims_c[3]);

    output_im2col->setZeroTensor();
    output_direct->setZeroTensor();
    output_im2winBASE->setZeroTensor();
    output_im2winHPC->setZeroTensor();

    Convolution<double>* conv_im2col = new Im2colConv<double>(input, filter, output_im2col, stride);
    Convolution<double>* conv_direct = new DirectConv<double>(input, filter, output_direct, stride);
    Convolution<double>* conv_im2winBASE = new Im2winConvBase<double>(input, filter, output_im2winBASE, stride);
    //Convolution<double>* conv_im2winHPC = new Im2winConvHPC<double>(input, filter, output_im2winHPC, stride);
    
    conv_im2col->conv_implement();
    conv_direct->conv_implement();
    conv_im2winBASE->conv_implement();
    //conv_im2winHPC->conv_implement();

    double max_diff_im2winConvBASE, max_diff_direct, max_diff_im2winConvHPC;
    max_diff_direct = output_im2col->compareTensor(*output_direct);
    max_diff_im2winConvBASE = output_im2col->compareTensor(*output_im2winBASE);
    //max_diff_im2winConvHPC = output_im2col->compareTensor(*output_im2winHPC);
    
    std::cout<<"max_diff_direct : "<<max_diff_direct<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"max_diff_im2winConvBASE : "<<max_diff_im2winConvBASE<<std::endl;
    std::cout<<"               "<<std::endl;
    //std::cout<<"max_diff_im2winConvHPC : "<<max_diff_im2winConvHPC<<std::endl;
    std::cout<<"               "<<std::endl;

    double timeIm2col = minIm2colConv(iter, input, filter, output_im2col, stride);
    double timeDirect = minDirectConv(iter, input, filter, output_direct, stride);
    double timeIm2winBASE = minIm2winBASEConv(iter, input, filter, output_im2winBASE, stride);
    //double timeIm2winHPC = minIm2winHPCConv(iter, input, filter, output_im2winHPC, stride);
    
    auto gflops = conv_im2col->get_gflops();

    double GFLOPS_Im2col = gflops/timeIm2col;
    double GFLOPS_Direct = gflops/timeDirect;
    double GFLOPS_Im2winBASE = gflops/timeIm2winBASE;
    //double GFLOPS_Im2winHPC = gflops/timeIm2winHPC;

    vector<double> log_output;
    std::cout<<"run_time_im2col is :"<<timeIm2col<<"s"<<std::endl;
    std::cout<<"run_time_direct is :"<<timeDirect<<"s"<<std::endl;
    std::cout<<"run_time_im2win is :"<<timeIm2winBASE<<"s"<<std::endl;
    //std::cout<<"run_time_im2winHPC is :"<<timeIm2winHPC<<"s"<<std::endl;
    std::cout<<"               "<<std::endl;
    std::cout<<"GFLOP is :"<<gflops<<std::endl;
    std::cout<<"GFLOPS_Im2col is :"<<GFLOPS_Im2col<<std::endl;
    std::cout<<"GFLOPS_Direct is :"<<GFLOPS_Direct<<std::endl;
    std::cout<<"GFLOPS_Im2winBASE is :"<<GFLOPS_Im2winBASE<<std::endl;
    //std::cout<<"GFLOPS_Im2winHPC is :"<<GFLOPS_Im2winHPC<<std::endl;

    //CLEANUP();
    
    return log_output;
}

template<class dataType>
vector<double> test_enter(size_t *dims_a, size_t *dims_b, size_t stride){
    size_t iter = 100;
    // WeTensor<dataType> *input, *filter, *output;
    dataType flag;
    vector<double> log_output =  test_implement(iter, dims_a, dims_b, stride, flag);
    return log_output;
}

template<class dataType>
vector<double> test_conv1(size_t batch){
    size_t dims_a[4] = {batch, 3, 227, 227};
    size_t dims_b[4] = {96, 3, 11, 11};
    size_t stride = 4;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv1-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv2(size_t batch){
    size_t dims_a[4] = {batch, 3, 231, 231};
    size_t dims_b[4] = {96, 3, 11, 11};
    size_t stride = 4;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv2-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv3(size_t batch){
    size_t dims_a[4] = {batch, 3, 227, 227};
    size_t dims_b[4] = {64, 3, 7, 7};
    size_t stride = 2;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv3-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv4(size_t batch){
    size_t dims_a[4] = {batch, 32, 224, 224};
    size_t dims_b[4] = {64, 32, 7, 7};
    size_t stride = 2;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv4-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv5(size_t batch){
    size_t dims_a[4] = {batch, 96, 24, 24};
    size_t dims_b[4] = {256, 96, 5, 5};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv5-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv6(size_t batch){
    size_t dims_a[4] = {batch, 256, 12, 12};
    size_t dims_b[4] = {512, 256, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv6-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv7(size_t batch){
    size_t dims_a[4] = {batch, 3, 224, 224};
    size_t dims_b[4] = {64, 3, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv7-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv8(size_t batch){
    size_t dims_a[4] = {batch, 64, 112, 112};
    size_t dims_b[4] = {128, 64, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv8-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv9(size_t batch){
    size_t dims_a[4] = {batch, 64, 56, 56};
    size_t dims_b[4] = {64, 64, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv9-----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv10(size_t batch){
    size_t dims_a[4] = {batch, 128, 28, 28};
    size_t dims_b[4] = {128, 128, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv10----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv11(size_t batch){
    size_t dims_a[4] = {batch, 256, 14, 14};
    size_t dims_b[4] = {256, 256, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv11----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
vector<double> test_conv12(size_t batch){
    size_t dims_a[4] = {batch, 512, 7, 7};
    size_t dims_b[4] = {512, 512, 3, 3};
    size_t stride = 1;
    std::cout<<"--------------------"<<std::endl;
    std::cout<<"-----test_conv12----"<<std::endl;
    vector<double> output = test_enter<dataType>(dims_a, dims_b, stride);
    std::cout<<"--------------------"<<std::endl;
    return output;
}

template<class dataType>
int test_benchmarks(){
    ofstream dataFile;
    string fileName;
    if(typeid(dataType) == typeid(float))
    fileName = fileName+"log_float_";
    else if(typeid(dataType) == typeid(double))
    fileName = fileName+"log_double_";

    time_t now = time(NULL);
    char chNow[256];
    strftime(chNow, sizeof(chNow), "%Y%m%d %H%M", localtime(&now));
    string stNow = chNow;
    fileName = fileName + stNow +".txt";
    dataFile.open(fileName, ofstream::app);
    fstream file(fileName, ios::out);

    size_t batch = 128;
    dataFile <<'\t'<<"im2c_t"<<'\t'<<"dir_t"<<'\t'<<"im2wBase_t"<<'\t'<<"im2wHPC_t"<<'\t'<<"im2c_G"<<'\t'<<"dir_G"<<'\t'<<"im2wBase_G"<<'\t'<<"im2w_G"<<endl;
    vector<double> outpout1 = test_conv1<dataType>(batch);
    dataFile<<"Conv1"<<'\t'<<outpout1[0]<<'\t'<<outpout1[1]<<'\t'<<outpout1[2]<<'\t'<<outpout1[3]<<'\t'<<outpout1[4]<<'\t'<<outpout1[5]<<'\t'<<outpout1[6]<<'\t'<<outpout1[7]<<endl;
    vector<double> outpout2 = test_conv2<dataType>(batch);
    dataFile<<"Conv2"<<'\t'<<outpout2[0]<<'\t'<<outpout2[1]<<'\t'<<outpout2[2]<<'\t'<<outpout2[3]<<'\t'<<outpout2[4]<<'\t'<<outpout2[5]<<'\t'<<outpout2[6]<<'\t'<<outpout2[7]<<endl;
    vector<double> outpout3 = test_conv3<dataType>(batch);
    dataFile<<"Conv3"<<'\t'<<outpout3[0]<<'\t'<<outpout3[1]<<'\t'<<outpout3[2]<<'\t'<<outpout3[3]<<'\t'<<outpout3[4]<<'\t'<<outpout3[5]<<'\t'<<outpout3[6]<<'\t'<<outpout3[7]<<endl;
    vector<double> outpout4 = test_conv4<dataType>(batch);
    dataFile<<"Conv4"<<'\t'<<outpout4[0]<<'\t'<<outpout4[1]<<'\t'<<outpout4[2]<<'\t'<<outpout4[3]<<'\t'<<outpout4[4]<<'\t'<<outpout4[5]<<'\t'<<outpout4[6]<<'\t'<<outpout4[7]<<endl;
    vector<double> outpout5 = test_conv5<dataType>(batch);
    dataFile<<"Conv5"<<'\t'<<outpout5[0]<<'\t'<<outpout5[1]<<'\t'<<outpout5[2]<<'\t'<<outpout5[3]<<'\t'<<outpout5[4]<<'\t'<<outpout5[5]<<'\t'<<outpout5[6]<<'\t'<<outpout5[7]<<endl;
    vector<double> outpout6 = test_conv6<dataType>(batch);
    dataFile<<"Conv6"<<'\t'<<outpout6[0]<<'\t'<<outpout6[1]<<'\t'<<outpout6[2]<<'\t'<<outpout6[3]<<'\t'<<outpout6[4]<<'\t'<<outpout6[5]<<'\t'<<outpout6[6]<<'\t'<<outpout6[7]<<endl;
    vector<double> outpout7 = test_conv7<dataType>(batch);
    dataFile<<"Conv7"<<'\t'<<outpout7[0]<<'\t'<<outpout7[1]<<'\t'<<outpout7[2]<<'\t'<<outpout7[3]<<'\t'<<outpout7[4]<<'\t'<<outpout7[5]<<'\t'<<outpout7[6]<<'\t'<<outpout7[7]<<endl;
    vector<double> outpout8 = test_conv8<dataType>(batch);
    dataFile<<"Conv8"<<'\t'<<outpout8[0]<<'\t'<<outpout8[1]<<'\t'<<outpout8[2]<<'\t'<<outpout8[3]<<'\t'<<outpout8[4]<<'\t'<<outpout8[5]<<'\t'<<outpout8[6]<<'\t'<<outpout8[7]<<endl;
    vector<double> outpout9 = test_conv9<dataType>(batch);
    dataFile<<"Conv9"<<'\t'<<outpout9[0]<<'\t'<<outpout9[1]<<'\t'<<outpout9[2]<<'\t'<<outpout9[3]<<'\t'<<outpout9[4]<<'\t'<<outpout9[5]<<'\t'<<outpout9[6]<<'\t'<<outpout9[7]<<endl;
    vector<double> outpout10 = test_conv10<dataType>(batch);
    dataFile<<"Conv10"<<'\t'<<outpout10[0]<<'\t'<<outpout10[1]<<'\t'<<outpout10[2]<<'\t'<<outpout10[3]<<'\t'<<outpout10[4]<<'\t'<<outpout10[5]<<'\t'<<outpout10[6]<<'\t'<<outpout10[7]<<endl;
    vector<double> outpout11 = test_conv11<dataType>(batch);
    dataFile<<"Conv11"<<'\t'<<outpout11[0]<<'\t'<<outpout11[1]<<'\t'<<outpout11[2]<<'\t'<<outpout11[3]<<'\t'<<outpout11[4]<<'\t'<<outpout11[5]<<'\t'<<outpout11[6]<<'\t'<<outpout11[7]<<endl;
    vector<double> outpout12 = test_conv12<dataType>(batch);
    dataFile<<"Conv12"<<'\t'<<outpout12[0]<<'\t'<<outpout12[1]<<'\t'<<outpout12[2]<<'\t'<<outpout12[3]<<'\t'<<outpout12[4]<<'\t'<<outpout12[5]<<'\t'<<outpout12[6]<<'\t'<<outpout12[7]<<endl;
    return 0;
}

#endif
