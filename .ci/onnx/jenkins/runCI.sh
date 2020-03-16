#!/bin/bash

# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

HELP_MESSAGE="Usage: runCI.sh [ARGS] [--help]

Builds nGraph and runs nGraph-ONNX tests.

Arguments:
    --help                      - Displays this message.
    --cleanup                   - Removes docker container and files created during script execution, instead of running CI.
    --rebuild                   - Rebuilds nGraph and runs tox tests.
    --ngraph-onnx-branch=...    - Will use specified branch of nGraph-ONNX.
                                  Default: master
    --ngraph-onnx-sha=...       - Checkout to specified nGraph-ONNX commit.
                                  Default: none - latest commit of cloned branch used
    --backends=...              - Comma separated list (no whitespaces!) of nGraph backends to run CI on.
                                  Default: cpu,interpreter
"

NGRAPH_ONNX_REPO_ADDRESS="https://github.com/NervanaSystems/ngraph-onnx.git"
NGRAPH_ONNX_REPO_DIR_NAME="ngraph-onnx"
NGRAPH_ONNX_CI_DIR=".ci/jenkins"

DEFAULT_NGRAPH_ONNX_REPO_BRANCH="master"
DEFAULT_NGRAPH_ONNX_REPO_SHA=""

# "<OPERATING_SYSTEM>" will be replaced by particular OS during script execution
DOCKER_CONTAINER_NAME_PATTERN="ngraph-onnx_ci_reproduction_<OPERATING_SYSTEM>"
DOCKER_IMAGE_NAME_PATTERN="aibt/aibt/ngraph/<OPERATING_SYSTEM>/base"
DOCKER_BASE_IMAGE_TAG="ci"
DOCKER_EXEC_IMAGE_TAG="ci_run"
DOCKER_HOME="/home/${USER}"

DEFAULT_BACKENDS="cpu interpreter"
HTTP_PROXY="${http_proxy}"
HTTPS_PROXY="${https_proxy}"

function main() {
    # Main function

    # Load parameters defaults
    CLEANUP="false"
    REBUILD="false"
    NGRAPH_ONNX_REPO_BRANCH="${DEFAULT_NGRAPH_ONNX_REPO_BRANCH}"
    NGRAPH_ONNX_REPO_SHA="${DEFAULT_NGRAPH_ONNX_REPO_SHA}"
    BACKENDS="${DEFAULT_BACKENDS}"

    parse_arguments "${@}"

    # Calculate necessary paths
    NGRAPH_CI_ABS_PATH="$(pwd)/$( dirname "${BASH_SOURCE[0]}" )"
    NGRAPH_ROOT_ABS_PATH="${NGRAPH_CI_ABS_PATH%/.ci*}"
    local ngraph_parent_path="$(dirname ${NGRAPH_ROOT_ABS_PATH})"
    WORKSPACE="${ngraph_parent_path}"
    NGRAPH_ONNX_REPO_PATH="${WORKSPACE}/${NGRAPH_ONNX_REPO_DIR_NAME}"
    NGRAPH_ONNX_CI_PATH="${NGRAPH_ONNX_REPO_DIR_NAME}/${NGRAPH_ONNX_CI_DIR}"

    cd "${WORKSPACE}"

    if [ "${CLEANUP}" = "true" ]; then
        cleanup
        return 0
    fi

    if ! check_ngraph_onnx_repo; then
        echo "[INFO] nGraph-ONNX repository is going to be cloned to ${NGRAPH_ONNX_REPO_PATH}"
        git clone "${NGRAPH_ONNX_REPO_ADDRESS}" --branch "${NGRAPH_ONNX_REPO_BRANCH}" "${NGRAPH_ONNX_REPO_PATH}"
    fi

    local cloned_repo_branch="$(ngraph_onnx_rev_parse "--abbrev-ref")"
    if [[ "${cloned_repo_branch}"!="${NGRAPH_ONNX_REPO_BRANCH}" ]]; then
        echo "[INFO] Checking out nGraph-ONNX to ${NGRAPH_ONNX_REPO_BRANCH}"
        checkout_ngraph_onnx_repo "${NGRAPH_ONNX_REPO_BRANCH}"
    fi

    local cloned_repo_sha="$(ngraph_onnx_rev_parse)"
    if [ -z "${NGRAPH_ONNX_REPO_SHA}" ]; then
        NGRAPH_ONNX_REPO_SHA="${cloned_repo_sha}"
    fi

    if [[ "${NGRAPH_ONNX_REPO_SHA}" != "${cloned_repo_sha}" ]]; then
        echo "[INFO] Checking out nGraph-ONNX to ${NGRAPH_ONNX_REPO_BRANCH}"
        checkout_ngraph_onnx_repo "${NGRAPH_ONNX_REPO_SHA}"
    fi

    run_ci

    return 0
}

function parse_arguments {
    # Parses script arguments
    PATTERN='[-a-zA-Z0-9_]*='
    for i in "${@}"; do
        case $i in
            "--help")
                printf ${HELP_MESSAGE}
                exit 0
                ;;
            "--cleanup")
                CLEANUP="true"
                echo "[INFO] Cleanup will be performed"
                ;;
            "--rebuild")
                REBUILD="true"
                echo "[INFO] nGraph is going to be rebuilt"
                ;;
            "--ngraph-onnx-branch="*)
                NGRAPH_ONNX_REPO_BRANCH="${i//${PATTERN}/}"
                ;;
            "--ngraph-onnx-sha="*)
                NGRAPH_ONNX_REPO_SHA="${i//${PATTERN}/}"
                echo "[INFO] Using nGraph-ONNX commit ${NGRAPH_ONNX_REPO_SHA}"
                ;;
            "--backends="*)
                BACKENDS="${i//${PATTERN}/}"
                # Convert comma separated values into whitespace separated
                BACKENDS="${BACKENDS//,/ }"
                ;;
            *)
                echo "[ERROR] Unrecognized argument: ${i}"
                printf ${HELP_MESSAGE}
                exit -1
                ;;
        esac
    done

    echo "[INFO] Using nGraph-ONNX branch ${NGRAPH_ONNX_REPO_BRANCH}"
    echo "[INFO] Backends tested: ${BACKENDS}"

    return 0
}

function cleanup() {
    # Performs cleanup of artifacts and containers from previous runs.
    local container_name_pattern="${DOCKER_CONTAINER_NAME_PATTERN/<OPERATING_SYSTEM>/*}"
    echo "[INFO] Performing cleanup"
    docker rm -f "$(docker ps -a --format="{{.ID}}" --filter="name=${container_name_pattern}")"
    rm -rf "${NGRAPH_ROOT_ABS_PATH}/python/dist/ngraph*.whl"
    rm -rf "${NGRAPH_ROOT_ABS_PATH}/python/*.so"
    rm -rf "${NGRAPH_ROOT_ABS_PATH}/python/build"
    rm -rf "${NGRAPH_ONNX_REPO_PATH}"

    return 0
}

function check_ngraph_onnx_repo() {
    # Verifies if nGraph-ONNX repository is present
    local ngraph_onnx_git="${NGRAPH_ONNX_REPO_PATH}/.git"
    if [ -d "${ngraph_onnx_git}" ]; then
        # 0 - true
        return 0
    else
        # 1 - false
        return 1
    fi
}

function ngraph_onnx_rev_parse() {
    # Returns the result of git rev-parse on nGraph-ONNX repository.
    local rev_parse_args="${1}"
    local previous_dir="$(pwd)"
    local ngraph_onnx_dir="${NGRAPH_ONNX_REPO_PATH}"
    cd "${ngraph_onnx_dir}"
    local result="$(git rev-parse ${rev_parse_args} HEAD)"
    cd "${previous_dir}"
    echo "${result}"

    return 0
}

function checkout_ngraph_onnx_repo() {
    # Switches nGraph-ONNX repository to commit SHA
    local rev="${1}"
    local previous_dir="$(pwd)"
    cd "${NGRAPH_ONNX_REPO_PATH}"
    git checkout "${rev}"
    cd "${previous_dir}"

    return 0
}

function run_ci() {
    # Builds necessary Docker images and executes CI
    for dockerfile in $(find ${WORKSPACE}/${NGRAPH_ONNX_CI_PATH}/dockerfiles -maxdepth 1 -name *.dockerfile -exec basename {} \;); do
        local operating_system="${dockerfile/.dockerfile/}"
        echo "[INFO] Running CI for operating system ${operating_system}"
        local docker_container_name="${DOCKER_CONTAINER_NAME_PATTERN/<OPERATING_SYSTEM>/$operating_system}"
        local docker_image_name="${DOCKER_IMAGE_NAME_PATTERN/<OPERATING_SYSTEM>/$operating_system}"
        # Rebuild container if REBUILD parameter used or if there's no container present
        if [[ "${REBUILD}" = "true" || -z "$(check_container_status "${docker_container_name}")" ]]; then
            docker rm -f "${docker_container_name}" >/dev/null 2>&1
            build_docker_image "${operating_system}" "${docker_image_name}"
            run_docker_container "${docker_image_name}" "${docker_container_name}"
            prepare_environment "${docker_container_name}"
        elif [[ "$(check_container_status)"==*"Exited"* ]]; then
            docker start "${docker_container_name}"
        fi
        run_tox_tests "${docker_container_name}"
    done

    return 0
}

function check_container_status() {
    # Returns status of container for container name given as parameter
    local docker_container_name="${1}"
    echo "$(docker ps -a --format="{{ .Status }}" --filter="name=${docker_container_name}")"

    return 0
}

function build_docker_image() {
    # Builds CI Docker image for operating system given as parameter
    local operating_system="${1}"
    local docker_image_name="${2}"
    local dockerfiles_dir="${WORKSPACE}/${NGRAPH_ONNX_CI_PATH}/dockerfiles"
    local postprocess_dockerfile_subpath="postprocess/append_user.dockerfile"
    echo "[INFO] Building base image"
    docker build \
        --build-arg http_proxy="${HTTP_PROXY}" \
        --build-arg https_proxy="${HTTPS_PROXY}" \
        -f "${dockerfiles_dir}/${operating_system}.dockerfile" \
        -t "${docker_image_name}:${DOCKER_BASE_IMAGE_TAG}" .
    echo "[INFO] Building CI execution image with appended user"
    docker build \
        --build-arg base_image="${docker_image_name}:${DOCKER_BASE_IMAGE_TAG}" \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        --build-arg USERNAME="${USER}" \
        -f "${dockerfiles_dir}/${postprocess_dockerfile_subpath}" \
        -t "${docker_image_name}:${DOCKER_EXEC_IMAGE_TAG}" .

    return 0
}

function run_docker_container() {
    # Runs Docker container using image specified as parameter
    local docker_image_name="${1}"
    local docker_container_name="${2}"
    echo "[INFO] Running Docker container ${docker_container_name}"
    docker run -td \
                --privileged \
                --user "${USER}" \
                --name "${docker_container_name}"  \
                --volume "${WORKSPACE}:${DOCKER_HOME}" \
                ${docker_image_name}:${DOCKER_EXEC_IMAGE_TAG}

    return 0
}

function prepare_environment() {
    # Prepares environment - builds nGraph
    local docker_container_name="${1}"
    echo "[INFO] Building nGraph in Docker container ${docker_container_name}"
    docker exec ${docker_container_name} bash -c "${DOCKER_HOME}/${NGRAPH_ONNX_CI_PATH}/prepare_environment.sh \
                                                    --build-dir=${DOCKER_HOME} \
                                                    --backends=${BACKENDS// /,}"

    return 0
}

function run_tox_tests() {
    # Executes tox tests for every backend
    local docker_container_name="${1}"
    for backend in ${BACKENDS}; do
        run_backend_test "${docker_container_name}" "${backend}"
    done

    return 0
}

function run_backend_test() {
    # Executes single set of tox tests for backend given as parameter
    local docker_container_name="${1}"
    local backend="${2}"
    local ngraph_repo_dir_name="${NGRAPH_ROOT_ABS_PATH#*$WORKSPACE/}"
    local backend_env="NGRAPH_BACKEND=$(printf '%s\n' "${backend}" | awk '{ print toupper($0) }')"
    local ngraph_whl=$(docker exec ${docker_container_name} find ${DOCKER_HOME}/${ngraph_repo_dir_name}/python/dist/ -name 'ngraph*.whl')
    local tox_env="TOX_INSTALL_NGRAPH_FROM=${ngraph_whl}"
    echo "[INFO] Running tox tests in Docker container ${docker_container_name} for ${backend} backend"
    docker exec -e "${tox_env}" -e "${backend_env}" -w "${DOCKER_HOME}/${NGRAPH_ONNX_REPO_DIR_NAME}" ${docker_container_name} tox -c .

    return 0
}

main "${@}"
