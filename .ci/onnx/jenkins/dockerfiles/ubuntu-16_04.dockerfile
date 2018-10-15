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
