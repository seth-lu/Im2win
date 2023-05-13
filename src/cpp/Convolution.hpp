#ifndef _CONVOLUTION_
#define _CONVOLUTION_

#include "WeTensor.hpp"
#include "im2winSIMD.hpp"
#include "../cu/convImplentCUDA.cuh"
#include "mkl.h"
//#include <cblas.h>

template<class dataType>
class Convolution{
    public:
    WeTensor<dataType> *input, *filter, *output;
    size_t stride;
    virtual long double get_gflops() = 0;
    virtual void conv_implement() = 0;
};

template<class dataType>
class Im2colConv :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    size_t stride;
    Im2colConv(WeTensor<dataType> *input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;};
    long double get_gflops() override;
    void conv_implement() override{
        if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){return;}
        if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){return;}
        output->getDataTensor() = at::conv2d(input->getDataTensor(), filter->getDataTensor(), {}, stride);
        return;};
};

template<class dataType>
long double Im2colConv<dataType>::get_gflops(){
    size_t filter_batch = filter->batch_size;
    size_t filter_channel = filter->channel;
    size_t filter_height = filter->height;
    size_t filter_width = filter->width;

    size_t output_batch = output->batch_size;
    size_t output_channel = output->channel;
    size_t output_height = output->height;
    size_t output_width = output->width;

    // std::cout<<"output_height :"<<output_height<<std::endl;
    // std::cout<<"output_width :"<<output_width<<std::endl;
    // std::cout<<"output_batch :"<<output_batch<<std::endl;
    // std::cout<<"output_channel :"<<output_channel<<std::endl;
    // std::cout<<"filter_channel :"<<filter_channel<<std::endl;
    // std::cout<<"filter_height :"<<filter_height<<std::endl;
    // std::cout<<"filter_width :"<<filter_width<<std::endl;
    //  
    long double gflops = 1e-9 * output_height * output_width * output_batch * output_channel * filter_channel * filter_height * filter_width * 2;
    // std::cout<<"gflops :"<<gflops<<std::endl;                          
    return gflops;
}

template<class dataType>
class DirectConv :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    size_t stride;
    DirectConv(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
    long double get_gflops() override;
    void conv_implement() override;
};

template<class dataType>
DirectConv<dataType>::DirectConv(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;
}

template<class dataType>
long double DirectConv<dataType>::get_gflops(){
     size_t filter_batch = filter->batch_size;
     size_t filter_channel = filter->channel;
     size_t filter_height = filter->height;
     size_t filter_width = filter->width;

     size_t output_batch = output->batch_size;
     size_t output_channel = output->channel;
     size_t output_height = output->height;
     size_t output_width = output->width;

    long double gflops = 1e-9 * output_height * output_width * output_batch 
                              * output_channel * filter_channel * filter_height * filter_width * 2;
    return gflops;
}

template<class dataType>
void DirectConv<dataType>::conv_implement(){
    dataType *inptr, *filptr, *outptr;
    inptr = input->getDataPtr();
    filptr = filter->getDataPtr();
    outptr = output->getDataPtr();

    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;

    const size_t output_batch = output->batch_size;
    const size_t output_channel = output->channel;
    const size_t output_height = output->height;
    const size_t output_width = output->width;    

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layout!"<<std::endl;
        return;
    }
    if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){
        std::cout<<"data not same device!"<<std::endl;
        return;
    }

    if(input->deviceStorage == cpu){
        if(input->dataLayout == NCHW){
            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(size_t i = 0; i < output_batch; i++){
                size_t iY = i * output_channel;
                size_t iD = i * input_channel;
                for(size_t j = 0; j < output_channel; j++){
                    size_t jiY = (iY + j) * output_height;
                    size_t jW = j * input_channel;
                    for(size_t k = 0; k < output_height; k++) {
                        size_t kjiY = (jiY + k) * output_width;
                        size_t kD =  k * input_width * stride_height;
                        for(size_t l = 0; l < output_width; l++){
                            size_t indexY = kjiY + l;
                            dataType tmp = *(outptr+indexY);
                            size_t headD = kD + l * stride_width;
                            for (size_t c = 0; c < input_channel; c++) {
                                size_t ciD = (iD + c) * input_height * input_width + headD;
                                size_t cjW = (jW + c) * filter_height;
                                for (size_t h = 0; h < filter_height; h++) {
                                    size_t hciD = ciD + h * input_width;
                                    size_t hcjW = (cjW + h) * filter_width;
                                    for (size_t w = 0; w < filter_width; w++) {
                                        size_t whcD = hciD + w;
                                        size_t whcW = hcjW + w;
                                        tmp += inptr[whcD] * filptr[whcW];
                                    }
                                }
                            }
                            *(outptr+indexY) = tmp;
                        }
                    }
                }
            }
        }
        else if(input->dataLayout == NHWC){
            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(size_t i = 0; i < output_batch; i++){
                size_t iY = i * output_height;
                size_t iD = i * input_height;
                for(size_t k = 0; k < output_height; k++){
                    size_t kiY = (iY + k) * output_width;
                    for(size_t l = 0; l < output_width; l++){
                        size_t lkiY = (kiY + l) * output_channel;
                        size_t lD =  l * input_width * stride_height;
                        for(size_t j = 0; j < output_channel; j++){
                            size_t indexY = lkiY + j;
                            size_t jW = j * filter_height;
                            dataType tmp = *(outptr+indexY);
                            size_t headD = k * input_channel * input_width * stride_height + l * stride_width * input_channel;
                            for (size_t h = 0; h < filter_height; h++) {

                                for (size_t w = 0; w < filter_width; w++) {

                                    for (size_t c = 0; c < input_channel; c++) {
                                        size_t whcD = (iD + h) * input_width * input_channel + headD  + w * input_channel + c;
                                        size_t whcW = (jW + h) * filter_width * input_channel + w * input_channel + c;
                                        tmp += inptr[whcD] * filptr[whcW];
                                    }
                                }
                            }
                            *(outptr+indexY) = tmp;
                        }
                    }
                }
            }
        }
    }
    else if(input->deviceStorage == gpu){
        size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
        size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
        size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};

        if(input->dataLayout == NCHW){
            directConvCUDAimplentNCHW(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
        else if(input->dataLayout == NHWC){
            implicitConvCUDAimplentNHWC(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
    }        
    return;
}

template<class dataType>
class Im2winConvBase :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    size_t stride;
    Im2winConvBase(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
    long double get_gflops() override;
    at::Tensor image2window();
    at::Tensor filter2window();
    void conv_implement() override;
};

template<class dataType>
Im2winConvBase<dataType>::Im2winConvBase(WeTensor<dataType> *input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;
}

template<class dataType>
long double Im2winConvBase<dataType>::get_gflops(){
     size_t filter_batch = filter->batch_size;
     size_t filter_channel = filter->channel;
     size_t filter_height = filter->height;
     size_t filter_width = filter->width;

     size_t output_batch = output->batch_size;
     size_t output_channel = output->channel;
     size_t output_height = output->height;
     size_t output_width = output->width;

    long double gflops = 1e-9 * output_height * output_width * output_batch 
                              * output_channel * filter_channel * filter_height * filter_width * 2;
    return gflops;
}

template<class dataType>
at::Tensor Im2winConvBase<dataType>::image2window(){
    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;  

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t window_row = (input_height - filter_height) / stride_height + 1;
    const size_t window_col = (input_width - filter_width) / stride_width + 1;

    at::Tensor output = at::empty({0}, input->getDataTensor().options());
   
    // size_t output_size;
    if(filter_height > stride_height && filter_width > stride_width)
        // output_size = input_batch * input_channel * window_row * input_width * filter_height;
        output.resize_({int(input_batch), int(input_channel), int(window_row), int(input_width * filter_height)});
    else
        // output_size = input_batch * input_channel * input_height * input_width;
        output.resize_({int(input_batch), int(input_channel), int(input_height), int(input_width)});
    // output.resize_(output_size);
    int layout_src = input->dataLayout;
    if(layout_src == NHWC)  
        input->contiguous();

    dataType *inptr, *outptr;
    inptr = input->getDataPtr();
    outptr = output.data_ptr<dataType>();

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < input_batch; ++i){ 
        size_t iY = i * input_channel;
        size_t iX = i * input_channel * input_height * input_width;
        for(size_t j = 0; j < input_channel; ++j){
            size_t jiY =  (iY + j) * window_row;
            size_t jiX =  iX + j * input_height * input_width;
            for(size_t k = 0; k < window_row; ++k){
                size_t kjiY = (jiY + k) * input_width;
                size_t kjiX = jiX + k * input_width * stride_height;
            for(size_t m = 0; m < input_width; ++m){
                size_t mkjiY = (kjiY + m) * filter_height;
                size_t mkjiX = kjiX + m;
                    for(size_t n = 0; n < filter_height; ++n){
                        size_t nmkjiY = mkjiY + n;
                        size_t nmkjiX = mkjiX  + n * input_width;
                        *(outptr + nmkjiY) = *(inptr + nmkjiX);
                    }
                }
            }
        }
    }
    if(layout_src == NHWC){
        input->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    return output;
}

template<class dataType>
at::Tensor Im2winConvBase<dataType>::filter2window(){
    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width; 

    at::Tensor output = at::empty({0}, filter->getDataTensor().options());
    // output.resize_(filter_batch * filter_channel * filter_height * filter_width);
    output.resize_({int(filter_batch), int(filter_channel), int(filter_height), int(filter_width)});

    int layout_src = input->dataLayout;
    if(layout_src == NHWC)  
        filter->contiguous();
    dataType *srcptr, *outptr;
    srcptr = filter->getDataPtr();
    outptr = output.data_ptr<dataType>();

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < filter_batch; i++){
        size_t iD = i * filter_channel;
        size_t iY = i * filter_channel * filter_height * filter_width; 
        for(size_t j = 0; j < filter_channel; j++){
            size_t jiD = (iD + j) * filter_height;
            size_t jiY = iY + j * filter_height * filter_width;
            for(size_t m = 0; m < filter_height; m++){
                size_t mjiD = (jiD + m) * filter_width;
                size_t mjiY = jiY + m;
               for(size_t n = 0; n < filter_width; n++){
                   size_t nmjiD = mjiD + n;
                   size_t nmjiY = mjiY + n * filter_height;
                    *(outptr + nmjiY) = *(srcptr + nmjiD);
                }
            }
        }
    }
    if(layout_src == NHWC){
        filter->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    return output;
}

template<class dataType>
void Im2winConvBase<dataType>::conv_implement(){
    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layout!"<<std::endl;
        return;
    }
    if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){
        std::cout<<"data not same device!"<<std::endl;
        return;
    }

    at::Tensor input_win = image2window();
    at::Tensor filter_win = filter2window();

    dataType *inptr, *filptr, *outptr;
    inptr = input_win.data_ptr<dataType>();
    filptr = filter_win.data_ptr<dataType>();
    outptr = output->getDataPtr();

    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;

    const size_t output_batch = output->batch_size;
    const size_t output_channel = output->channel;
    const size_t output_height = output->height;
    const size_t output_width = output->width;    

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t output_csize = output_batch * output_channel * output_height * output_width;
    const size_t output_area = output_height * output_width;
    const size_t filter_area = filter_height * filter_width;
    const size_t filter_volume = filter_area * filter_channel;
    const size_t gap_width = stride_width * filter_height;
    const size_t window_row = filter_height * input_width;
    const size_t window_area = output_height * window_row;
    const size_t window_volume = input_channel * window_area;
    const size_t output_volume = filter_batch * output_area;

    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layerout!"<<std::endl;
        std::cout<<input->dataLayout<<std::endl;
        std::cout<<filter->dataLayout<<std::endl;
        std::cout<<output->dataLayout<<std::endl;
        return;
    }
    if(input->dataLayout == NCHW){
        #ifdef omp_flag
        #pragma omp parallel for schedule(omp_flag)
        #endif
        for(int b = 0; b < input_batch; ++b){
            int bD = b * window_volume;
            int bY = b * output_volume;
            for(int  c = 0; c < input_channel; ++c){
                int cbD = bD + c * window_area;
                int cW = c * filter_area;
                for(int n = 0; n < filter_batch; ++n){
                    int headW = n * filter_volume + cW;
                    int nbY = bY + n * output_area;
                    for(int i = 0; i < output_height; ++i){
                        int iD = i * window_row + cbD;
                        int inbY = nbY + i * output_width;
                        for(int j = 0; j < output_width; ++j){
                            int indexY = inbY + j;
                            dataType tmp = *(outptr+indexY);
                            int headD = iD + j * gap_width;
                            for(int w = 0; w < filter_width; ++w){
                                int wcD = w * filter_height + headD;
                                int wcW = w * filter_height + headW;
                                for(int h = 0; h < filter_height; ++h){
                                    int indexD = wcD + h;
                                    int indexW = wcW + h; 
                                    tmp += inptr[indexD] * filptr[indexW];
                                }
                            }
                            outptr[indexY] = tmp;
                        }
                    }
                }
            }
        }
    }
    else if(input->dataLayout == NHWC){
        #ifdef omp_flag
        #pragma omp parallel for schedule(omp_flag)
        #endif
        for(int b = 0; b < input_batch; ++b){
            int bY = b * output_height * output_width * filter_batch;
            int bD = b * input_channel * output_height * filter_height * input_width;
            for(int i = 0; i < output_height; ++i){
                int ibY = bY + i * output_width * filter_batch;
                int iD = i * filter_height * input_width * input_channel;
                for(int j = 0; j < output_width; ++j){
                    int jibY = ibY + j * filter_batch;
                    int headD = iD + j * stride_width * filter_height * input_channel;
                    for(int n = 0; n < filter_batch; ++n){
                        int indexY = jibY + n;
                        int headW = n * filter_height * filter_width * filter_channel;
                        dataType tmp = 0;
                        for(int h = 0; h < filter_height; ++h){    
                            int hbD = bD + h * input_channel;
                            int hW = h * input_channel;
                            for(int w = 0; w < filter_width; ++w){
                               int whbD = hbD + w * filter_height * input_channel;
                               int whW = hW + w * filter_height * input_channel;
                                for(int  c = 0; c < input_channel; ++c){
                                    int indexD = whbD + c + headD;
                                    int indexW =  whW + c + headW; 
                                    tmp += inptr[indexD] * filptr[indexW];
                                }
                            }
                        }
                        outptr[indexY] = tmp;
                    }
                }
            }
        }
    }
}

template<class dataType>
class Im2winConvBASE :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    at::Tensor input_win, filter_win;
    size_t stride;
    Im2winConvBASE(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
    ~Im2winConvBASE(){input_win.detach(); input_win.resize_(at::IntArrayRef{0}); filter_win.detach(); filter_win.resize_(at::IntArrayRef{0});};
    long double get_gflops() override;
    at::Tensor image2window();
    at::Tensor filter2window();
    void conv_implement() override;
};

template<class dataType>
Im2winConvBASE<dataType>::Im2winConvBASE(WeTensor<dataType> *input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;
    // std::cout<<"flag 1"<<std::endl;
    input_win = image2window();
    filter_win = filter2window();
    // std::cout<<"flag 2"<<std::endl;
    if(input->deviceStorage == gpu){
        input_win.cuda();
        filter_win.cuda();
    }
}

template<class dataType>
long double Im2winConvBASE<dataType>::get_gflops(){
     size_t filter_batch = filter->batch_size;
     size_t filter_channel = filter->channel;
     size_t filter_height = filter->height;
     size_t filter_width = filter->width;

     size_t output_batch = output->batch_size;
     size_t output_channel = output->channel;
     size_t output_height = output->height;
     size_t output_width = output->width;

    long double gflops = 1e-9 * output_height * output_width * output_batch 
                              * output_channel * filter_channel * filter_height * filter_width * 2;
    return gflops;
}

template<class dataType>
at::Tensor Im2winConvBASE<dataType>::image2window(){
    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;  

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t window_row = (input_height - filter_height) / stride_height + 1;
    const size_t window_col = (input_width - filter_width) / stride_width + 1;

    int layout_src = input->dataLayout;
    int device_src = input->deviceStorage;

    if(layout_src == NHWC)  
        input->contiguous();
    if(device_src == gpu)
        input->deviceToCPU();

    at::Tensor output = at::zeros({0}, input->getDataTensor().options());
   
    // size_t output_size;
    int input_channel_;
    if (input_channel < 8) input_channel_ = 8;
    else input_channel_ = input_channel;
   
    // size_t output_size;
    if(filter_height > stride_height && filter_width > stride_width)
        // output_size = input_batch * input_channel * window_row * input_width * filter_height;
        output.resize_({int(input_batch), int(input_channel), int(window_row), int(input_width * filter_height)});
    else
        // output_size = input_batch * input_channel * input_height * input_width;
        output.resize_({int(input_batch), int(input_channel), int(input_height), int(input_width)});
    // output.resize_(output_size);
    
    dataType *inptr, *outptr;
    inptr = input->getDataPtr();
    outptr = output.data_ptr<dataType>();
    // std::cout<<"before im2win run"<<std::endl;

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < input_batch; ++i){ 
        size_t iY = i * input_channel;
        size_t iX = i * input_channel * input_height * input_width;
        for(size_t j = 0; j < input_channel; ++j){
            size_t jiY =  (iY + j) * window_row;
            size_t jiX =  iX + j * input_height * input_width;
            for(size_t k = 0; k < window_row; ++k){
                size_t kjiY = (jiY + k) * input_width;
                size_t kjiX = jiX + k * input_width * stride_height;
            for(size_t m = 0; m < input_width; ++m){
                size_t mkjiY = (kjiY + m) * filter_height;
                size_t mkjiX = kjiX + m;
                    for(size_t n = 0; n < filter_height; ++n){
                        size_t nmkjiY = mkjiY + n;
                        size_t nmkjiX = mkjiX  + n * input_width;
                        *(outptr + nmkjiY) = *(inptr + nmkjiX);
                    }
                }
            }
        }
    }
    // std::cout<<"after im2win run"<<std::endl;
    if(layout_src == NHWC){
        input->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    if(device_src == gpu){
        input->deviceToGPU();
        output = output.cuda();
    }
    return output;
}

template<class dataType>
at::Tensor Im2winConvBASE<dataType>::filter2window(){
    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width; 

    int layout_src = input->dataLayout;
    int device_src = input->deviceStorage;
    if(layout_src == NHWC)  
        filter->contiguous();
    if(device_src == gpu)
        filter->deviceToCPU();

    at::Tensor output = at::zeros({0}, filter->getDataTensor().options());
    int filter_channel_;
    if (filter_channel < 8) filter_channel_ = 8;
    else filter_channel_ = filter_channel;
    // output.resize_(filter_batch * filter_channel * filter_height * filter_width);
    output.resize_({int(filter_batch), int(filter_channel), int(filter_height), int(filter_width)});    
    dataType *srcptr, *outptr;
    srcptr = filter->getDataPtr();
    outptr = output.data_ptr<dataType>();

    // std::cout<<"fliter copy start"<<std::endl;

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < filter_batch; i++){
        size_t iD = i * filter_channel;
        size_t iY = i * filter_channel * filter_height * filter_width; 
        for(size_t j = 0; j < filter_channel; j++){
            size_t jiD = (iD + j) * filter_height;
            size_t jiY = iY + j * filter_height * filter_width;
            for(size_t m = 0; m < filter_height; m++){
                size_t mjiD = (jiD + m) * filter_width;
                size_t mjiY = jiY + m;
               for(size_t n = 0; n < filter_width; n++){
                   size_t nmjiD = mjiD + n;
                   size_t nmjiY = mjiY + n * filter_height;
                    *(outptr + nmjiY) = *(srcptr + nmjiD);
                }
            }
        }
    }

    // std::cout<<"fliter copy end"<<std::endl;

    if(layout_src == NHWC){
        filter->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    if(device_src == gpu){
        filter->deviceToGPU();
        output = output.cuda();
    }

    return output;
}

template<class dataType>
void Im2winConvBASE<dataType>::conv_implement(){
    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layout!"<<std::endl;
        return;
    }
    if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){
        std::cout<<"data not same device!"<<std::endl;
        return;
    }

    dataType *inptr, *filptr, *outptr, tmp;
    inptr = input_win.data_ptr<dataType>();
    filptr = filter_win.data_ptr<dataType>();
    outptr = output->getDataPtr();

    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;

    const size_t output_batch = output->batch_size;
    const size_t output_channel = output->channel;
    const size_t output_height = output->height;
    const size_t output_width = output->width;    

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t output_csize = output_batch * output_channel * output_height * output_width;
    const size_t output_area = output_height * output_width;
    const size_t filter_area = filter_height * filter_width;
    const size_t filter_volume = filter_area * filter_channel;
    const size_t gap_width = stride_width * filter_height;
    const size_t window_row = filter_height * input_width;
    const size_t window_area = output_height * window_row;
    const size_t window_volume = input_channel * window_area;
    const size_t output_volume = filter_batch * output_area;

    if(input->deviceStorage == cpu){
        if(input->dataLayout == NCHW){
            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(int b = 0; b < input_batch; ++b){
                int bD = b * window_volume;
                int bY = b * output_volume;
                for(int  c = 0; c < input_channel; ++c){
                    int cbD = bD + c * window_area;
                    int cW = c * filter_area;
                    for(int n = 0; n < filter_batch; ++n){
                        int headW = n * filter_volume + cW;
                        int nbY = bY + n * output_area;
                        for(int i = 0; i < output_height; ++i){
                            int iD = i * window_row + cbD;
                            int inbY = nbY + i * output_width;
                            for(int j = 0; j < output_width; ++j){
                                int indexY = inbY + j;
                                dataType tmp = *(outptr+indexY);
                                int headD = iD + j * gap_width;
                                for(int w = 0; w < filter_width; ++w){
                                    int wcD = w * filter_height + headD;
                                    int wcW = w * filter_height + headW;
                                    for(int h = 0; h < filter_height; ++h){
                                        int indexD = wcD + h;
                                        int indexW = wcW + h; 
                                        tmp += inptr[indexD] * filptr[indexW];
                                    }
                                }
                                outptr[indexY] = tmp;
                            }
                        }
                    }
                }
            }
        }
        else if(input->dataLayout == NHWC){
            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(int b = 0; b < input_batch; ++b){
                int bY = b * output_height * output_width * filter_batch;
                int bD = b * input_channel * output_height * filter_height * input_width;
                for(int i = 0; i < output_height; ++i){
                    int ibY = bY + i * output_width * filter_batch;
                    int iD = i * filter_height * input_width * input_channel;
                    for(int j = 0; j < output_width; ++j){
                        int jibY = ibY + j * filter_batch;
                        int headD = iD + j * stride_width * filter_height * input_channel;
                        for(int n = 0; n < filter_batch; ++n){
                            int indexY = jibY + n;
                            int headW = n * filter_height * filter_width * filter_channel;
                            dataType tmp = 0;
                            for(int h = 0; h < filter_height; ++h){    
                                int hbD = bD + h * input_channel;
                                int hW = h * input_channel;
                                for(int w = 0; w < filter_width; ++w){
                                int whbD = hbD + w * filter_height * input_channel;
                                int whW = hW + w * filter_height * input_channel;
                                    for(int  c = 0; c < input_channel; ++c){
                                        int indexD = whbD + c + headD;
                                        int indexW =  whW + c + headW; 
                                        tmp += inptr[indexD] * filptr[indexW];
                                    }
                                }
                            }
                            outptr[indexY] = tmp;
                        }
                    }
                }
            }
        }
    }
    else if(input->deviceStorage == gpu){
        if(input->dataLayout == NCHW){
            size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
            size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
            size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};
            // std::cout<<"before test_cublas"<<std::endl;
            im2winConvCUDAimplentNCHWBASE(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
        else if(input->dataLayout == NHWC){
            size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
            size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
            size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};
            // std::cout<<"before test_cublas"<<std::endl;
            im2winConvCUDAimplentNHWCBASE(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
    }
}

template<class dataType>
class Im2winConvHPC :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    at::Tensor input_win, filter_win;
    size_t stride;
    Im2winConvHPC(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
    ~Im2winConvHPC(){input_win.detach(); input_win.resize_(at::IntArrayRef{0}); filter_win.detach(); filter_win.resize_(at::IntArrayRef{0});};
    long double get_gflops() override;
    at::Tensor image2window();
    at::Tensor filter2window();
    void conv_implement() override;
};

template<class dataType>
Im2winConvHPC<dataType>::Im2winConvHPC(WeTensor<dataType> *input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;

    // if (input->channel < 8){
    //     input->setSize(input->batch_size, 8, input->height, input->width);
    //     auto pad_input = at::zeros({int(input->batch_size), int(8 - input->channel), int(input->height), int(input->width)}, input->getDataTensor().options());
    //     input->dataTensor = at::cat({input->getDataTensor(), pad_input}, 1);
    //     input->setDataPtr();

    //     filter->setSize(filter->batch_size, 8, filter->height, filter->width);
    //     auto pad_filter = at::zeros({int(filter->batch_size), int(8 - filter->channel), int(filter->height), int(filter->width)}, filter->getDataTensor().options());
    //     filter->dataTensor = at::cat({filter->getDataTensor(), pad_filter}, 1);
    //     filter->setDataPtr();
    // }
    // printf("input size : %d, %d, %d, %d\n", input->batch_size,input->channel,input->height,input->width);
    // std::cout<<"flag 1"<<std::endl;
    input_win = image2window();
    filter_win = filter2window();
    // std::cout<<"flag 2"<<std::endl;
    if(input->deviceStorage == gpu){
        input_win.cuda();
        filter_win.cuda();
    }
}

template<class dataType>
long double Im2winConvHPC<dataType>::get_gflops(){
     size_t filter_batch = filter->batch_size;
     size_t filter_channel = filter->channel;
     size_t filter_height = filter->height;
     size_t filter_width = filter->width;

     size_t output_batch = output->batch_size;
     size_t output_channel = output->channel;
     size_t output_height = output->height;
     size_t output_width = output->width;

    long double gflops = 1e-9 * output_height * output_width * output_batch 
                              * output_channel * filter_channel * filter_height * filter_width * 2;
    return gflops;
}

template<>
at::Tensor Im2winConvHPC<float>::image2window(){
    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;  

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t window_row = (input_height - filter_height) / stride_height + 1;
    const size_t window_col = (input_width - filter_width) / stride_width + 1;

    int layout_src = input->dataLayout;
    int device_src = input->deviceStorage;

    if(layout_src == NHWC)  
        input->contiguous();
    if(device_src == gpu)
        input->deviceToCPU();

    at::Tensor output = at::zeros({0}, input->getDataTensor().options());
   
    // size_t output_size;
    int input_channel_;
    if (input_channel < 8) input_channel_ = 8;
    else input_channel_ = input_channel;

    if(filter_height > stride_height && filter_width > stride_width)
        // output_size = input_batch * input_channel * window_row * input_width * filter_height;
        output.resize_({int(input_batch), int(input_channel_), int(window_row), int(input_width * filter_height)});
    else
        // output_size = input_batch * input_channel * input_height * input_width;
        output.resize_({int(input_batch), int(input_channel_), int(input_height), int(input_width)});
    // output.resize_(output_size);
    
    float *inptr, *outptr;
    inptr = input->getDataPtr();
    outptr = output.data_ptr<float>();
    // std::cout<<"before im2win run"<<std::endl;

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < input_batch; ++i){ 
        size_t iY = i * input_channel;
        size_t iX = i * input_channel * input_height * input_width;
        for(size_t j = 0; j < input_channel; ++j){
            size_t jiY =  (iY + j) * window_row;
            size_t jiX =  iX + j * input_height * input_width;
            for(size_t k = 0; k < window_row; ++k){
                size_t kjiY = (jiY + k) * input_width;
                size_t kjiX = jiX + k * input_width * stride_height;
                for(size_t m = 0; m < input_width; ++m){
                    size_t mkjiY = (kjiY + m) * filter_height;
                    size_t mkjiX = kjiX + m;
                    for(size_t n = 0; n < filter_height; ++n){
                        size_t nmkjiY = mkjiY + n;
                        size_t nmkjiX = mkjiX  + n * input_width;
                        *(outptr + nmkjiY) = *(inptr + nmkjiX);
                    }
                }
            }
        }
    }
    // std::cout<<"after im2win run"<<std::endl;
    if(layout_src == NHWC){
        input->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    if(device_src == gpu){
        input->deviceToGPU();
        output = output.cuda();
    }
    return output;
}

template<>
at::Tensor Im2winConvHPC<float>::filter2window(){
    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width; 

    int layout_src = input->dataLayout;
    int device_src = input->deviceStorage;
    if(layout_src == NHWC)  
        filter->contiguous();
    if(device_src == gpu)
        filter->deviceToCPU();

    at::Tensor output = at::zeros({0}, filter->getDataTensor().options());
    int filter_channel_;
    if (filter_channel < 8) filter_channel_ = 8;
    else filter_channel_ = filter_channel;
    // output.resize_(filter_batch * filter_channel * filter_height * filter_width);
    output.resize_({int(filter_batch), int(filter_channel_), int(filter_height), int(filter_width)});    
    float *srcptr, *outptr;
    srcptr = filter->getDataPtr();
    outptr = output.data_ptr<float>();

    // std::cout<<"fliter copy start"<<std::endl;

    #ifdef omp_flag
    #pragma omp parallel for schedule(omp_flag)
    #endif
    for(size_t i = 0; i < filter_batch; i++){
        size_t iD = i * filter_channel;
        size_t iY = i * filter_channel * filter_height * filter_width; 
        for(size_t j = 0; j < filter_channel; j++){
            size_t jiD = (iD + j) * filter_height;
            size_t jiY = iY + j * filter_height * filter_width;
            for(size_t m = 0; m < filter_height; m++){
                size_t mjiD = (jiD + m) * filter_width;
                size_t mjiY = jiY + m;
               for(size_t n = 0; n < filter_width; n++){
                   size_t nmjiD = mjiD + n;
                   size_t nmjiY = mjiY + n * filter_height;
                    *(outptr + nmjiY) = *(srcptr + nmjiD);
                }
            }
        }
    }

    // std::cout<<"fliter copy end"<<std::endl;

    if(layout_src == NHWC){
        filter->channelsLast();  
        output = output.to(at::MemoryFormat::ChannelsLast);}
    if(device_src == gpu){
        filter->deviceToGPU();
        output = output.cuda();
    }

    return output;
}

template<>
void Im2winConvHPC<float>::conv_implement(){
    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layout!"<<std::endl;
        return;
    }
    if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){
        std::cout<<"data not same device!"<<std::endl;
        return;
    }

    float *inptr, *filptr, *outptr, tmp;
    inptr = input_win.data_ptr<float>();
    filptr = filter_win.data_ptr<float>();
    outptr = output->getDataPtr();

    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;

    const size_t output_batch = output->batch_size;
    const size_t output_channel = output->channel;
    const size_t output_height = output->height;
    const size_t output_width = output->width;    

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    const size_t output_csize = output_batch * output_channel * output_height * output_width;
    const size_t output_area = output_height * output_width;
    const size_t filter_area = filter_height * filter_width;
    const size_t filter_volume = filter_area * filter_channel;
    const size_t gap_width = stride_width * filter_height;
    const size_t window_row = filter_height * input_width;
    const size_t window_area = output_height * window_row;
    const size_t window_volume = input_channel * window_area;
    const size_t output_volume = filter_batch * output_area;

    if(input->deviceStorage == cpu){
        if(input->dataLayout == NCHW){
            int m, n, lda, incx, incy;
            float alpha, beta, *a, *x, *y;
            m = filter_batch;
            n = filter_area;
            lda = n;
            incx = 1;
            incy = output_height * output_width;
            alpha = 1;
            beta = 1;

            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(int b = 0; b < input_batch; ++b){
                int bY = b * output_volume;
                int bD = b * window_volume;
                for(int i = 0; i < output_height; ++i){
                    int ibY = bY + i * output_width;
                    int ibD = bD + i * window_row;
                    for(int  c = 0; c < input_channel; ++c){
                        int cibD = ibD + c * window_area;
                        int cW = c * filter_batch * filter_area;
                        for(int j = 0; j < output_width; ++j){
                            int jibY = ibY + j;
                            int jcibD = cibD + j * gap_width;

                            a = filptr + cW;
                            x = inptr + jcibD;
                            y = outptr + jibY;
                            //mkl_enable_instructions(MKL_ENABLE_AVX2);
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda,
                                        x, incx, beta, y, incy);
                            // cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda,
                            //             x+1*gap_width, incx, beta, y+1, incy);
                            // j += 2;
                            // for(int n = 0; n < filter_batch; ++n){
                            //     int ncW = cW + n * filter_area;
                            //     int indexY = jibY + n * output_area;
                            //     tmp = 0; 
                            //     for(int w = 0; w < filter_width; ++w){
                            //         int wcjibD = cjibD + w * filter_height;
                            //         int wcW = ncW + w * filter_height;
                            //         for(int h = 0; h < filter_height; ++h){
                            //             int indexD = wcjibD + h;
                            //             int indexW = wcW + h; 
                            //             tmp += *(inptr + indexD) * (*(filptr + indexW));
                            //             // std::cout<<*(inptr + indexD)<<"-"<<*(filptr + indexW)<<" ";
                            //         }
                            //         //*(outptr + indexY) += cblas_sdot(filter_height, inptr + wcD, 1, filptr + wcW, 1); //lower performance than src
                            //     }
                            //     *(outptr + indexY) += tmp;
                            // }
                        }
                    }
                }
            }
        }
        else if(input->dataLayout == NHWC){
            int m, n, lda, incx, incy;
            float alpha, beta, *a, *x, *y;
            m = filter_batch;
            n = filter_height * filter_width * input_channel;
            lda = n;
            incx = 1;
            incy = 1;
            alpha = 1;
            beta = 1;
            #ifdef omp_flag
            #pragma omp parallel for schedule(omp_flag)
            #endif
            for(int b = 0; b < input_batch; ++b){
                int bY = b * output_height * output_width * filter_batch;
                int bD = b * output_height * filter_height * input_width * input_channel;
                for(int i = 0; i < output_height; ++i){
                    int ibY = bY + i * output_width * filter_batch;
                    int ibD = bD + i * filter_height * input_width * input_channel;
                        for(int j = 0; j < output_width; ++j){
                            int jibY = ibY + j * filter_batch;
                            int jcibD = ibD + j * stride_width * filter_height * input_channel;

                            a = filptr;
                            x = inptr + jcibD;
                            y = outptr + jibY;
                            //mkl_enable_instructions(MKL_ENABLE_AVX2);
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda,
                                        x, incx, beta, y, incy);
                            // cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda,
                            //             x+1*gap_width, incx, beta, y+1, incy);
                            // j += 2;
                            // for(int n = 0; n < filter_batch; ++n){
                            //     int ncW = cW + n * filter_area;
                            //     int indexY = jibY + n * output_area;
                            //     tmp = 0; 
                            //     for(int w = 0; w < filter_width; ++w){
                            //         int wcjibD = cjibD + w * filter_height;
                            //         int wcW = ncW + w * filter_height;
                            //         for(int h = 0; h < filter_height; ++h){
                            //             int indexD = wcjibD + h;
                            //             int indexW = wcW + h; 
                            //             tmp += *(inptr + indexD) * (*(filptr + indexW));
                            //             // std::cout<<*(inptr + indexD)<<"-"<<*(filptr + indexW)<<" ";
                            //         }
                            //         //*(outptr + indexY) += cblas_sdot(filter_height, inptr + wcD, 1, filptr + wcW, 1); //lower performance than src
                            //     }
                            //     *(outptr + indexY) += tmp;
                            // }
                        }
                    
                }
            }
        }
    }
    else if(input->deviceStorage == gpu){
        if(input->dataLayout == NCHW){
            size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
            size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
            size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};
            // std::cout<<"before test_cublas"<<std::endl;
            im2winConvCUDAimplentNCHWHPC(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
        else if(input->dataLayout == NHWC){
            size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
            size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
            size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};
            // std::cout<<"before test_cublas"<<std::endl;
            //im2winConvCUDAimplentNHWCHPC(inptr, filptr, outptr, dims_a, dims_b, dims_c);
            im2winConvCUDAimplentNHWCBLAS(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
    }
}
// template<class dataType>
// class Im2winConvSIMD :public Convolution<dataType>{
//     public:
//     WeTensor<dataType> *input, *filter, *output;
//     size_t stride;
//     at::Tensor input_win, filter_win;
//     at::Tensor image2window();
//     at::Tensor filter2window();
//     Im2winConvSIMD(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
//     long double get_gflops() override;
//     void conv_implement() override;
// };

// template<class dataType>
// Im2winConvSIMD<dataType>::Im2winConvSIMD(WeTensor<dataType> *input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
//     input = input_;
//     filter = filter_;
//     output = output_;
//     stride = stride_;
//     input_win = image2window();
//     filter_win = filter2window();
// }

// template<class dataType>
// long double Im2winConvSIMD<dataType>::get_gflops(){
//      size_t filter_batch = filter->batch_size;
//      size_t filter_channel = filter->channel;
//      size_t filter_height = filter->height;
//      size_t filter_width = filter->width;

//      size_t output_batch = output->batch_size;
//      size_t output_channel = output->channel;
//      size_t output_height = output->height;
//      size_t output_width = output->width;

//     long double gflops = 1e-9 * output_height * output_width * output_batch 
//                               * output_channel * filter_channel * filter_height * filter_width * 2;
//     return gflops;
// }

// template<class dataType>
// at::Tensor Im2winConvSIMD<dataType>::image2window(){
//     const size_t input_batch = input->batch_size;
//     const size_t input_channel = input->channel;
//     const size_t input_height = input->height;
//     const size_t input_width = input->width;

//     const size_t filter_batch = filter->batch_size;
//     const size_t filter_channel = filter->channel;
//     const size_t filter_height = filter->height;
//     const size_t filter_width = filter->width;  

//     const size_t stride_height = stride;
//     const size_t stride_width = stride;

//     const size_t window_row = (input_height - filter_height) / stride_height + 1;
//     const size_t window_col = (input_width - filter_width) / stride_width + 1;

//     at::Tensor output = at::empty({0}, input->getDataTensor().options());
   
//     // size_t output_size;
//     if(filter_height > stride_height && filter_width > stride_width)
//         // output_size = input_batch * input_channel * window_row * input_width * filter_height;
//         output.resize_({int(input_batch), int(input_channel), int(window_row), int(input_width * filter_height)});
//     else
//         // output_size = input_batch * input_channel * input_height * input_width;
//         output.resize_({int(input_batch), int(input_channel), int(input_height), int(input_width)});
//     // output.resize_(output_size);
//     int layout_src = input->dataLayout;
//     if(layout_src == NHWC)  
//         input->contiguous();

//     dataType *inptr, *outptr;
//     inptr = input->getDataPtr();
//     outptr = output.data_ptr<dataType>();

//     #ifdef omp_flag
//     #pragma omp parallel for schedule(omp_flag)
//     #endif
//     for(size_t i = 0; i < input_batch; ++i){ 
//         size_t iY = i * input_channel;
//         size_t iX = i * input_channel * input_height * input_width;
//         for(size_t j = 0; j < input_channel; ++j){
//             size_t jiY =  (iY + j) * window_row;
//             size_t jiX =  iX + j * input_height * input_width;
//             for(size_t k = 0; k < window_row; ++k){
//                 size_t kjiY = (jiY + k) * input_width;
//                 size_t kjiX = jiX + k * input_width * stride_height;
//             for(size_t m = 0; m < input_width; ++m){
//                 size_t mkjiY = (kjiY + m) * filter_height;
//                 size_t mkjiX = kjiX + m;
//                     for(size_t n = 0; n < filter_height; ++n){
//                         size_t nmkjiY = mkjiY + n;
//                         size_t nmkjiX = mkjiX  + n * input_width;
//                         *(outptr + nmkjiY) = *(inptr + nmkjiX);
//                     }
//                 }
//             }
//         }
//     }
//     if(layout_src == NHWC){
//         input->channelsLast();  
//         output = output.to(at::MemoryFormat::ChannelsLast);}
//     return output;
// }

// template<class dataType>
// at::Tensor Im2winConvSIMD<dataType>::filter2window(){
//     const size_t filter_batch = filter->batch_size;
//     const size_t filter_channel = filter->channel;
//     const size_t filter_height = filter->height;
//     const size_t filter_width = filter->width; 

//     at::Tensor output = at::empty({0}, filter->getDataTensor().options());
//     // output.resize_(filter_batch * filter_channel * filter_height * filter_width);
//     output.resize_({int(filter_batch), int(filter_channel), int(filter_height), int(filter_width)});

//     int layout_src = input->dataLayout;
//     if(layout_src == NHWC)  
//         filter->contiguous();
//     dataType *srcptr, *outptr;
//     srcptr = filter->getDataPtr();
//     outptr = output.data_ptr<dataType>();

//     #ifdef omp_flag
//     #pragma omp parallel for schedule(omp_flag)
//     #endif
//     for(size_t i = 0; i < filter_batch; i++){
//         size_t iD = i * filter_channel;
//         size_t iY = i * filter_channel * filter_height * filter_width; 
//         for(size_t j = 0; j < filter_channel; j++){
//             size_t jiD = (iD + j) * filter_height;
//             size_t jiY = iY + j * filter_height * filter_width;
//             for(size_t m = 0; m < filter_height; m++){
//                 size_t mjiD = (jiD + m) * filter_width;
//                 size_t mjiY = jiY + m;
//                for(size_t n = 0; n < filter_width; n++){
//                    size_t nmjiD = mjiD + n;
//                    size_t nmjiY = mjiY + n * filter_height;
//                     *(outptr + nmjiY) = *(srcptr + nmjiD);
//                 }
//             }
//         }
//     }
//     if(layout_src == NHWC){
//         filter->channelsLast();  
//         output = output.to(at::MemoryFormat::ChannelsLast);}
//     return output;
// }

// template<class dataType>
// void Im2winConvSIMD<dataType>::conv_implement(){
//     if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
//         std::cout<<"data not same layout!"<<std::endl;
//         return;}
//     dataType *inptr, *filptr, *outptr;
//     inptr = input_win.data_ptr<dataType>();
//     filptr = filter_win.data_ptr<dataType>();
//     outptr = output->getDataPtr();

//     size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
//     size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
//     size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};
//     if(input->dataLayout == NCHW)
//         IM2WIN_CONV_SIMD(inptr, filptr, outptr, dims_a, dims_b, dims_c);
//     else if(input->dataLayout == NHWC)
//     std::cout<<"IM2WIN_SIMD_NHWC pass"<<std::endl;
//         // IM2WIN_CONV_SIMD_AVX512(inptr, filptr, outptr, dims_a, dims_b, dims_c);
// }


template<class dataType>
class cuDNNConv :public Convolution<dataType>{
    public:
    WeTensor<dataType> *input, *filter, *output;
    size_t stride;
    cuDNNConv(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_);
    long double get_gflops() override;
    void conv_implement() override;
};

template<class dataType>
cuDNNConv<dataType>::cuDNNConv(WeTensor<dataType>* input_, WeTensor<dataType> *filter_, WeTensor<dataType> *output_, size_t stride_){
    input = input_;
    filter = filter_;
    output = output_;
    stride = stride_;
}

template<class dataType>
long double cuDNNConv<dataType>::get_gflops(){
     size_t filter_batch = filter->batch_size;
     size_t filter_channel = filter->channel;
     size_t filter_height = filter->height;
     size_t filter_width = filter->width;

     size_t output_batch = output->batch_size;
     size_t output_channel = output->channel;
     size_t output_height = output->height;
     size_t output_width = output->width;

    long double gflops = 1e-9 * output_height * output_width * output_batch 
                              * output_channel * filter_channel * filter_height * filter_width * 2;
    return gflops;
}

template<class dataType>
void cuDNNConv<dataType>::conv_implement(){
    dataType *inptr, *filptr, *outptr;
    inptr = input->getDataPtr();
    filptr = filter->getDataPtr();
    outptr = output->getDataPtr();

    const size_t input_batch = input->batch_size;
    const size_t input_channel = input->channel;
    const size_t input_height = input->height;
    const size_t input_width = input->width;

    const size_t filter_batch = filter->batch_size;
    const size_t filter_channel = filter->channel;
    const size_t filter_height = filter->height;
    const size_t filter_width = filter->width;

    const size_t output_batch = output->batch_size;
    const size_t output_channel = output->channel;
    const size_t output_height = output->height;
    const size_t output_width = output->width;    

    const size_t stride_height = stride;
    const size_t stride_width = stride;

    if(!(input->dataLayout == filter->dataLayout && filter->dataLayout == output->dataLayout)){
        std::cout<<"data not same layout!"<<std::endl;
        return;
    }
    if(!(input->deviceStorage == filter->deviceStorage && filter->deviceStorage == output->deviceStorage)){
        std::cout<<"data not same device!"<<std::endl;
        return;
    }

    if(input->deviceStorage == cpu){
        std::cout<<"data should on gpu!"<<std::endl;
        return;
    }
    else if(input->deviceStorage == gpu){
        size_t dims_a[4]={input->batch_size, input->channel, input->height, input->width};
        size_t dims_b[4]={filter->batch_size, filter->channel, filter->height, filter->width};
        size_t dims_c[4]={output->batch_size, output->channel, output->height, output->width};

        if(input->dataLayout == NCHW){
            cuDNNConvNCHW(inptr, filptr, outptr, dims_a, dims_b, dims_c);
        }
        else if(input->dataLayout == NHWC){
            std::cout<<"data should NCHW!"<<std::endl;
        }
    }        
    return;
}

#endif
