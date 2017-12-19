# ngraph-neon

## Installation

Follow these steps to install the ngraph's python wrapper and its prerequisites.

### Pybind

Download the required version of pybind and install it.
```
git clone -b allow-nonconstructible-holders https://github.com/jagerman/pybind11
cd pybind11
mkdir build && cd build
cmake ..
make -j4
make install
cd ../
pip install -e .
```
### private-ngraph-cpp

Download the required version of private-ngraph-cpp and install it.
```
git clone https://github.com/NervanaSystems/private-ngraph-cpp.git
cd private-ngraph-cpp
git checkout 3b84d91a5819045bb74387eb965c57f6058483a9 -b "local branch name"
Build and Install it : https://github.com/NervanaSystems/private-ngraph-cpp#steps
```

### ngraph-neon

After installing pybind and private-ngraph-cpp, follow the steps below to install ngraph-neon.
The NGRAPH_CPP_BUILD_PATH is set to default installation location of private-ngraph-cpp.
```
git clone https://github.com/NervanaSystems/ngraph-neon.git
cd ngraph-neon
export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist/
export PYBIND_HEADERS_PATH="Path to Pybind headers"
pip install -e .
```
