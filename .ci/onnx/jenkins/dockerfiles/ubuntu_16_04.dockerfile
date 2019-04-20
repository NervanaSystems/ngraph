FROM ubuntu:16.04

ARG HOME=/root
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# nGraph dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  clang-3.9 \
  git \
  curl \
  zlib1g \
  zlib1g-dev \
  libtinfo-dev \
  unzip \
  autoconf \
  automake \
  libtool && \
  apt-get clean autoclean && apt-get autoremove -y

# Python dependencies
RUN apt-get -y install python3 \
                       python3-pip \
                       python3-dev \
                       python-virtualenv && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip3 install --upgrade pip setuptools wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Install tox
RUN pip3 install tox

# Build nGraph master
ARG NGRAPH_CACHE_DIR=/cache

WORKDIR /root

RUN git clone https://github.com/NervanaSystems/ngraph.git && \
    cd ngraph && \
    mkdir -p ./build && \
    cd ./build && \
    cmake ../ -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE && \
    make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)

# Store built nGraph
RUN mkdir -p ${NGRAPH_CACHE_DIR} && \
    cp -Rf /root/ngraph/build ${NGRAPH_CACHE_DIR}/

# Cleanup remaining sources
RUN rm -rf /root/ngraph
