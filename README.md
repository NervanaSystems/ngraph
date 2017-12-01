# ngraph-neon

## Installation

Follow these steps to install the ngraph's python wrapper and its prerequisites.

### Pybind

Download the required version of pybind and install it.
```
git clone https://github.com/jagerman/pybind11
cd pybind11
git checkout 53be81931f35313f70affc9826bdfed9820cce2c
mkdir build && cd build
cmake ..
make -j4
make install
cd ../
pip install -e .
```