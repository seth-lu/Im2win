#include "WeTensor.hpp"
#include "Convolution.hpp"
#include "test_benchmark.cpp"

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

int main(){
    test_benchmarks<float>();

    return 0;
}