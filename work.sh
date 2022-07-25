rm -rf build
mkdir build
cd build
cmake .. -DUSE_MKL=OFF
make