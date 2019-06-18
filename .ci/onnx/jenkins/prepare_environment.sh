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

set -x
set -e

NGRAPH_CACHE_DIR="/cache"

function build_ngraph() {
    set -x
    # directory containing ngraph repo
    local ngraph_directory="$1"
    local func_parameters="$2"
    cd "${ngraph_directory}/ngraph"
    for parameter in $func_parameters
    do
        case $parameter in
            REBUILD)
                rm -rf "${ngraph_directory}/ngraph/build"
                rm -rf "${ngraph_directory}/ngraph_dist"
            ;;
            USE_CACHED)
                cp -Rf "${NGRAPH_CACHE_DIR}/build" "${ngraph_directory}/ngraph/" || return 1
            ;;
        esac
    done
    cd "${ngraph_directory}/ngraph"
    mkdir -p ./build
    cd ./build
    cmake ../ -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DCMAKE_INSTALL_PREFIX="${ngraph_directory}/ngraph_dist" || return 1
    make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l) || return 1
    make install || return 1
    cd "${ngraph_directory}/ngraph/python"
    if [ ! -d ./pybind11 ]; then
        git clone --recursive https://github.com/pybind/pybind11.git
    fi
    rm -f "${ngraph_directory}"/ngraph/python/dist/ngraph*.whl
    rm -rf "${ngraph_directory}/ngraph/python/*.so ${ngraph_directory}/ngraph/python/build"
    export PYBIND_HEADERS_PATH="${ngraph_directory}/ngraph/python/pybind11"
    export NGRAPH_CPP_BUILD_PATH="${ngraph_directory}/ngraph_dist"
    export NGRAPH_ONNX_IMPORT_ENABLE="TRUE"
    python3 setup.py bdist_wheel || return 1
    # Clean build artifacts
    rm -rf "${ngraph_directory}/ngraph_dist"
    return 0
}

function main() {
    # By default copy stored nGraph master and use it to build PR branch
    BUILD_CALL='build_ngraph "/root" "USE_CACHED" || build_ngraph "/root" "REBUILD"'

    PATTERN='[-a-zA-Z0-9_]*='
    for i in "$@"
    do
        case $i in
            --no-incremental)
                # Build nGraph from scratch if incremental building is disabled
                BUILD_CALL='build_ngraph "/root"'
                ;;
            *)
                echo "Parameter $i not recognized."
                exit 1
                ;;
        esac
    done

    # Link Onnx models
    mkdir -p /home/onnx_models/.onnx
    ln -s /home/onnx_models/.onnx /root/.onnx

    eval "${BUILD_CALL}"
}

if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    main "${@}"
fi
