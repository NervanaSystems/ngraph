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
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NGRAPH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../.. && pwd )"
BUILD_DIR="$( pwd )"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
    DOCKER_RUN_HTTP_PROXY="-e http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
    DOCKER_RUN_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
    DOCKER_RUN_HTTPS_PROXY="-e https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
    DOCKER_RUN_HTTPS_PROXY=' '
fi


# build the docker base image
docker build  --rm=true \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
       -f="${SCRIPT_DIR}/docker/Dockerfile.ngraph.manylinux2010" \
       -t="ngraph:manylinux2010" \
       ${SCRIPT_DIR}

# build manulinux1 wheels
docker run -it -u`id -u`:`id -g` \
       -v ${NGRAPH_DIR}:${NGRAPH_DIR} \
       -v `pwd`:`pwd` -w `pwd` \
       ${DOCKER_RUN_HTTP_PROXY} ${DOCKER_RUN_HTTPS_PROXY} \
       ngraph:manylinux2010 \
       ${SCRIPT_DIR}/helper/run-unit-test-helper.sh
