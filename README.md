# ngraph-neon

## Installation

Follow these steps to install the ngraph's python wrapper and its prerequisites.

### Pybind

Download the required version of pybind and install it.
```
git clone https://github.com/jagerman/pybind11
cd pybind11
mkdir build && cd build
cmake ..
make check -j 4
```