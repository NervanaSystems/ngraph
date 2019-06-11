FROM ubuntu:16.04

ARG HOME=/root
ARG NGRAPH_INSTALL_DIR=${HOME}/ngraph_build
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# copies all files and directories from the Docker build context
# !important - the build context needs to contain a directory with drivers named "intel-opencl"
COPY . ${HOME}

# nGraph dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
        build-essential=12.1ubuntu2 \
        cmake=3.5.1-1ubuntu3 \
        clang-3.9=1:3.9.1-4ubuntu3~16.04.2 \
        git=1:2.7.4-0ubuntu1.6 \
        curl=7.47.0-1ubuntu2.13 \
        zlib1g=1:1.2.8.dfsg-2ubuntu4.1 \
        zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1 \
        libtinfo-dev=6.0+20160213-1ubuntu1 \
        unzip=6.0-20ubuntu1 \
        autoconf=2.69-9 \
        automake=1:1.15-4ubuntu1 \
        ocl-icd-opencl-dev \
        libtool=2.4.6-0.1 && \
  apt-get clean autoclean && \
  apt-get autoremove -y

# install the iGPU drivers copied into the container from the build context
WORKDIR ${HOME}/intel-opencl
RUN dpkg -i *

# Python dependencies
RUN apt-get -y --no-install-recommends install \
        python3=3.5.1-3 \
        python3-pip=8.1.1-2ubuntu0.4 \
        python3-dev=3.5.1-3 \
        python-virtualenv=15.0.1+ds-3ubuntu1 && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip3 install --upgrade pip==19.0.3 \
        setuptools==41.0.0 \
        wheel==0.33.1

# ONNX dependencies
RUN apt-get -y --no-install-recommends install \
        protobuf-compiler=2.6.1-1.3 \
        libprotobuf-dev=2.6.1-1.3 && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
RUN git clone --depth=1 https://github.com/NervanaSystems/ngraph.git && \
    git clone --depth=1 https://github.com/NervanaSystems/ngraph-onnx.git && \
    mkdir -p ${HOME}/ngraph/build && \
    mkdir ${NGRAPH_INSTALL_DIR}

ENV LD_LIBRARY_PATH=${NGRAPH_INSTALL_DIR}/lib
ENV NGRAPH_CPP_BUILD_PATH=${NGRAPH_INSTALL_DIR}
ENV NGRAPH_ONNX_IMPORT_ENABLE=TRUE 
ENV PYBIND_HEADERS_PATH=${HOME}/ngraph/python/pybind11

WORKDIR ${HOME}/ngraph/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=${NGRAPH_INSTALL_DIR} \
    -DNGRAPH_TOOLS_ENABLE=FALSE \
    -DNGRAPH_UNIT_TEST_ENABLE=FALSE \
    -DNGRAPH_USE_PREBUILT_LLVM=TRUE \
    -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE \
    -DNGRAPH_WARNINGS_AS_ERRORS=TRUE \
    -DCMAKE_BUILD_TYPE=Release \
    -DNGRAPH_INTELGPU_ENABLE=TRUE && \
    make install -j "$(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)"

RUN cd ${HOME}/ngraph/python && \
    git clone --depth=1 --recursive https://github.com/jagerman/pybind11.git && \
    python3 setup.py develop

WORKDIR ${HOME}/ngraph-onnx

RUN pip3 install -r requirements.txt && \
    pip3 install -r requirements_test.txt

WORKDIR ${HOME}/ngraph-onnx/tests

