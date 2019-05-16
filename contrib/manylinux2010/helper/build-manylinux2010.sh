#!/bin/bash
# ******************************************************************************
# Copyright 2019 Intel Corporation
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cmake -DNGRAPH_PYTHON_BUILD_ENABLE=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_MANYLINUX_ENABLE=TRUE -DNGRAPH_DEX_ONLY=TRUE ${SCRIPT_DIR}/../../..
lcores=$(grep processor /proc/cpuinfo | wc -l)
make -j$lcores manylinux_wheel
