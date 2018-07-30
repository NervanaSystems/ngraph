FROM ubuntu:16.04

# nGraph dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  clang-3.9 \
  clang-format-3.9 \
  git \
  curl \
  zlib1g \
  zlib1g-dev \
  libtinfo-dev && \
  apt-get clean autoclean && apt-get autoremove -y

# Python dependencies
RUN apt-get update && apt-get install -y \
  python \
  python3 \
  python-dev \
  python3-dev \
  python-pip && \
  apt-get clean autoclean && apt-get autoremove -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install tox

# Prepare nGraph
COPY . /root/ngraph
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DCMAKE_INSTALL_PREFIX="~/ngraph_dist"
RUN make style-check
RUN make 
RUN make install

# Prepare nGraph Python API
WORKDIR /root/ngraph/python
RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
ENV NGRAPH_CPP_BUILD_PATH /root/ngraph_dist
ENV LD_LIBRARY_PATH /root/ngraph_dist/lib
ENV PYBIND_HEADERS_PATH /root/ngraph/python/pybind11

RUN df -k 

# Test nGraph and nGraph Python API
CMD cd /root/ngraph/build && make check ; cd /root/ngraph/python && tox
