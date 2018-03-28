#!/bin/sh
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
set -e

if [ -d build ]; then
    rm -rf build
fi

mkdir build

cd build

if [ -z "${NGRAPH_CPP_BUILD_PATH+x}" ]; then
  echo "NGRAPH_CPP_BUILD_PATH is not set"
  cmake ..
else
  echo "NGRAPH_CPP_BUILD_PATH set to" $NGRAPH_CPP_BUILD_PATH
  cmake -DNGRAPH_INSTALL_PREFIX=$NGRAPH_CPP_BUILD_PATH ..
fi

make -j8

cd ..
python2 setup.py bdist_wheel
