#include "../src/cpp/WeTensor.hpp"
#include "../src/cpp/Convolution.hpp"
#include "benchmark.h"
#include <cuda_runtime.h>
#include <stdlib.h>

int test_conv_num(int num, int batch){
    switch (num)
    {
    case 1:
        test_conv1<float>(batch);
        break;
    case 2:
        test_conv2<float>(batch);
        break;
    case 3:
        test_conv3<float>(batch);
        break;
    case 4:
        test_conv4<float>(batch);
        break;
    case 5:
        test_conv5<float>(batch);
        break;
    case 6:
        test_conv6<float>(batch);
        break;
    case 7:
        test_conv7<float>(batch);
        break;
    case 8:
        test_conv8<float>(batch);
        break;
    case 9:
        test_conv9<float>(batch);
        break;
    case 10:
        test_conv10<float>(batch);
        break;
    case 11:
        test_conv11<float>(batch);
        break;
    case 12:
        test_conv12<float>(batch);
        break;
                                                                                                
    }
    return 0;
}

int seleclt_test(int argc, char *argv[]){

    //std::cout<<"CUDA::is_available(): "<<torch::cuda::is_available()<<std::endl;
    char *convNum;
    if (argc != 1){
        convNum = argv[1];
        printf("test conv num: %d\n",atoi(convNum));
    }

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {

    std::cerr << "Error resetting GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    int batch = 128;
    test_conv_num(atoi(convNum), batch);
    return 0;
}

int main(int argc, char *argv[]){
    std::cout<<torch::show_config()<<std::endl;
    //seleclt_test(argc, argv);
    test_benchmarks<float>();
    return 0;
}
