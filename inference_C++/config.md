#NATIVE COMPILING ON LINUX ARM DEVICE
#Easy, slower

#Docker build runs on a Raspberry Pi 3B with Raspbian Stretch Lite OS (Desktop version will run #out memory when linking the .so file) will take 8-9 hours in total.

#You should folowing the steps below to build the onnxruntime.

sudo apt-get update
sudo apt-get install -y \
    sudo \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    tar

pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade wheel
pip3 install numpy

# Build the latest cmake
mkdir /code
cd /code
wget https://cmake.org/files/v3.13/cmake-3.23.1.tar.gz;
tar zxf cmake-3.23.1.tar.gz

cd /code/cmake-3.23.1
./configure --system-curl
make
sudo make install

# Prepare onnxruntime Repo
cd /code
git clone --recursive https://github.com/Microsoft/onnxruntime

# Start the basic build
cd /code/onnxruntime
./build.sh --config MinSizeRel --update --build

# Build Shared Library
./build.sh --config MinSizeRel --build_shared_lib

# Build Python Bindings and Wheel
./build.sh --config MinSizeRel --enable_pybind --build_wheel

# Build Output(*.so for C++ inference / *.whl for python inference)
ls -l /code/onnxruntime/build/Linux/MinSizeRel/*.so
ls -l /code/onnxruntime/build/Linux/MinSizeRel/dist/*.whl

# Use pip to install the .whl if needed.
pip install *.whl


