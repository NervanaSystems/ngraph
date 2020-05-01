#!/bin/bash
# ==============================================================================
#  Copyright 2017-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

mkdir build

cd build
if [ ${?} -ne 0 ]; then
    exit 1
fi

/usr/local/bin/cmake .. -DNGRAPH_CPU_ENABLE=1 -DNGRAPH_INTERPRETER_ENABLE=1 -DNGRAPH_GENERIC_CPU_ENABLE=0
if [ ${?} -ne 0 ]; then
    exit 1
fi

make -j64
if [ ${?} -ne 0 ]; then
    exit 1
fi

NGRAPH_CODEGEN=1 test/unit-test --gtest_filter=-INTERPRETER*
if [ ${?} -ne 0 ]; then
    exit 1
fi
