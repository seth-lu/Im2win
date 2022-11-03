#ifndef _WETENSOR_
#define _WETENSOR_

#include "ATen/ATen.h"
#include "torch/torch.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"
#include <stdio.h>
#include <omp.h>
#include <cmath>
#include <chrono>
#define omp_flag dynamic

enum layout{NCHW = 1, NHWC = 2};
enum device{cpu = 1, gpu = 2};

template<class dataType>
class WeTensor{
    public:
    at::Tensor dataTensor;
    size_t batch_size, channel, height, width;
    dataType *dataPtr;
    layout dataLayout;
    device deviceStorage;

    WeTensor(){};
    ~WeTensor(){dataTensor.detach(); dataTensor.resize_(at::IntArrayRef{0});};
    at::Tensor getDataTensor(){return dataTensor;};
    void setSize(size_t b, size_t c, size_t h, size_t w){batch_size=b, channel=c, height=h, width=w;return;};
    int* getSize();
    void setDataPtr(){dataPtr = dataTensor.data_ptr<dataType>(); return;};
    dataType* getDataPtr(){return dataPtr;};
    double compareTensor(WeTensor<dataType> &B);
    void channelsLast(){dataTensor = dataTensor.to(at::MemoryFormat::ChannelsLast); dataLayout = NHWC; setDataPtr(); return;};
    void contiguous(){dataTensor = dataTensor.to(at::MemoryFormat::Contiguous); dataLayout = NCHW; setDataPtr(); return;};
    void deviceToGPU(){dataTensor = dataTensor.cuda(); deviceStorage = gpu; setDataPtr(); return;}
    void deviceToCPU(){dataTensor = dataTensor.cpu(); deviceStorage = cpu; setDataPtr(); return;}

    virtual void initDataTensor() = 0;
    virtual void genArrangeTensor() = 0;
    virtual void setZeroTensor() = 0; 
};

template<class dataType>
double WeTensor<dataType>::compareTensor(WeTensor<dataType> &B){
    if(batch_size==B.batch_size&&channel==B.channel&&height==B.height&&width==B.width){
    double max_diff = -1;
    double diff = 1.0;
    dataType *a, *b;
    size_t iter = batch_size * channel * height * width;
    int device_src = deviceStorage;
    if(device_src == gpu){
        deviceToCPU();
        B.deviceToCPU();
    }
    a = getDataPtr();
    b = B.getDataPtr();
    for(size_t i = 0; i < iter; ++i){
        diff = std::abs(*(a + i) - *(b + i));
        max_diff = ( diff > max_diff ? diff : max_diff );
        }
    if(device_src == gpu){
        deviceToGPU();
        B.deviceToGPU();
    }
    return max_diff;
    }
    else{
        std::cout<<"The two tensors size not equal!"<<std::endl;
        return -1;
    }
}

class STensor :public WeTensor<float>{
    public:
    STensor(size_t b, size_t c, size_t h, size_t w);
    ~STensor(){dataTensor.detach(); dataTensor.resize_(at::IntArrayRef{0});};
    void initDataTensor() override;
    void genArrangeTensor() override;
    // double compareTensor(STensor &B) override;
    void setZeroTensor() override;
};

class DTensor :public WeTensor<double>{
    public:
    DTensor(size_t b, size_t c, size_t h, size_t w);
    void initDataTensor() override;
    void genArrangeTensor() override;
    // double compareTensor(DTensor &B) override;
    void setZeroTensor() override;
};

STensor::STensor(size_t b, size_t c, size_t h, size_t w){
    setSize(b, c, h, w);
}

DTensor::DTensor(size_t b, size_t c, size_t h, size_t w){
    setSize(b, c, h, w);
}

void STensor::initDataTensor(){
    dataTensor = torch::randn({int(batch_size), int(channel), int(height), int(width)}, torch::kFloat);
    dataLayout = NCHW;
    deviceStorage = cpu;
    setDataPtr();
    return;
}

void DTensor::initDataTensor(){
    dataTensor = torch::randn({int(batch_size), int(channel), int(height), int(width)}, torch::kDouble);
    dataLayout = NCHW;
    deviceStorage = cpu;
    setDataPtr();
    return;
}

void STensor::genArrangeTensor(){
    initDataTensor();
    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < batch_size; i++)
        for(size_t j = 0; j < channel; j++)
            for(size_t m = 0; m < height; m++)
                for(size_t n = 0; n < width; n++)
                    dataTensor[i][j][m][n] = float(i * channel * height * width
                                           + j * height * width + m * width + n+1);
    return;
}

void DTensor::genArrangeTensor(){
    initDataTensor();
    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < batch_size; i++)
        for(size_t j = 0; j < channel; j++)
            for(size_t m = 0; m < height; m++)
                for(size_t n = 0; n < width; n++)
                    dataTensor[i][j][m][n] = double(i * channel * height * width
                                           + j * height * width + m * width + n+1);
    return;
}

// double STensor::compareTensor(STensor &B){
//     if(batch_size==B.batch_size&&channel==B.channel&&height==B.height&&width==B.width){
//     static double max_diff = -1;
//     double diff = 1.0;
//     float *a, *b;
//     size_t iter = batch_size * channel * height * width;
//     a = getDataPtr();
//     b = B.getDataPtr();
//     for(size_t i = 0; i < iter; ++i){
//         diff = std::abs(*(a + i) - *(b + i));
//         max_diff = ( diff > max_diff ? diff : max_diff );
//         }
//     return max_diff;
//     }
//     else{
//         std::cout<<"The two tensors size not equal!"<<std::endl;
//         return -1;
//     }
// }

// double DTensor::compareTensor(DTensor &B){
//     if(batch_size==B.batch_size&&channel==B.channel&&height==B.height&&width==B.width){
//     static double max_diff = -1;
//     double diff = 1.0;
//     double *a, *b;
//     size_t iter = batch_size * channel * height * width;
//     a = getDataPtr();
//     b = B.getDataPtr();
//     for(size_t i = 0; i < iter; ++i){
//         diff = std::abs(*(a + i) - *(b + i));
//         max_diff = ( diff > max_diff ? diff : max_diff );
//         }
//     return max_diff;
//     }
//     else{
//         std::cout<<"The two tensors size not equal!"<<std::endl;
//         return -1;
//     }
// }

void STensor::setZeroTensor(){
    dataTensor = torch::zeros({int(batch_size), int(channel), int(height), int(width)}, torch::kFloat);
    dataLayout = NCHW;
    deviceStorage = cpu;
    setDataPtr();
    return;
}

void DTensor::setZeroTensor(){
    dataTensor = torch::zeros({int(batch_size), int(channel), int(height), int(width)}, torch::kDouble);
    dataLayout = NCHW;
    deviceStorage = cpu;
    setDataPtr();
    return;
}

#endif