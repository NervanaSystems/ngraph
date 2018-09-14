#!/bin/bash

# ******************************************************************************
# Copyright 2018 Intel Corporation
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

CI_PATH="$(pwd)"
CI_ROOT=".ci/jenkins"
REPO_ROOT="${CI_PATH%$CI_ROOT}"
DOCKER_CONTAINER="ngraph-onnx_ci"

# Function run() builds image with requirements needed to build ngraph and run onnx tests, runs container and executes tox tests
function run() {
    cd ./dockerfiles
    docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f=./ubuntu-16_04.dockerfile -t ngraph-onnx:ubuntu-16_04 .

    cd "${CI_PATH}"
    mkdir -p ${HOME}/ONNX_CI
    if [[ -z $(docker ps -a | grep -i "${DOCKER_CONTAINER}") ]]; 
    then 
        docker run -h "$(hostname)" --privileged --name "${DOCKER_CONTAINER}" -v ${HOME}/ONNX_CI:/home -v "${REPO_ROOT}":/root -d ngraph-onnx:ubuntu-16_04 tail -f /dev/null
        docker cp ./prepare_environment.sh "${DOCKER_CONTAINER}":/home
        docker exec "${DOCKER_CONTAINER}" ./home/prepare_environment.sh
    fi

    NGRAPH_WHL=$(docker exec ${DOCKER_CONTAINER} find /home/ngraph/python/dist/ -name "ngraph*.whl")
    docker exec -e TOX_INSTALL_NGRAPH_FROM="${NGRAPH_WHL}" "${DOCKER_CONTAINER}" tox -c /root

    echo "========== FOLLOWING ITEMS WERE CREATED DURING SCRIPT EXECUTION =========="
    echo "Docker image: ngraph-onnx:ubuntu-16_04"
    echo "Docker container: ${DOCKER_CONTAINER}"
    echo "Directory: ${HOME}/ONNX_CI"
    echo "Multiple files generated during tox execution"
    echo ""
    echo "TO REMOVE THEM RUN THIS SCRIPT WITH PARAMETER: --cleanup"
}

# Function cleanup() removes items created during script execution
function cleanup() {
    docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf /home/$(find /home/ -user root)'
    rm -rf ${HOME}/ONNX_CI
    docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf /root/$(find /root/ -user root)'
    docker rm -f "${DOCKER_CONTAINER}"
    docker rmi --force ngraph-onnx:ubuntu-16_04
}

PATTERN='[-a-zA-Z0-9_]*='
for i in "$@"
do
    case $i in
        --help*)
            printf "Following parameters are available:
    
            --help  displays this message
            --cleanup  removes docker image, container and files created during script execution
            "
            exit 0
        ;;
        --cleanup*)
            cleanup
            exit 0
        ;;
    esac
done

set -x

run