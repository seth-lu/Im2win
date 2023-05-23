rm -rf build
mkdir build
cd build
cmake -D CMAKE_PREFIX_PATH=~/libtorch ../benchmarks # Modify to the absolute path of ‘libtorch’
make
