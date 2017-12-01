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