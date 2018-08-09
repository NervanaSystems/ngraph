#!/bin/bash
# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related to the
# source code ("Material") are owned by Intel Corporation or its suppliers or
# licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and
# licensors, and is protected by worldwide copyright and trade secret laws and
# treaty provisions. No part of the Material may be used, copied, reproduced,
# modified, published, uploaded, posted, transmitted, distributed, or disclosed
# in any way without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery of
# the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.

# Install nGraph in /root/ngraph
cd /root
mkdir -p ./build
cd ./build
cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DCMAKE_INSTALL_PREFIX=/root/ngraph_dist
make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)
make install

# Build nGraph wheel
cd /root/ngraph/python
git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git

export PYBIND_HEADERS_PATH="/root/ngraph/python/pybind11"
export NGRAPH_CPP_BUILD_PATH="/root/ngraph_dist"
python3 setup.py bdist_wheel