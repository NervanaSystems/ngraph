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
readonly PARAMETERS=( 'name' 'version' 'container_name' 'volumes' 'env' 'ports' 'dockerfile_path' 'directory' 'options' 'tag' 'engine' 'frontend' 'new_tag' 'image_name' 'repository_type' 'build_cores_number')
readonly WORKDIR="$(git rev-parse --show-toplevel)"
readonly HUB_ADDRESS="hub.docker.intel.com"

#Example of usage: login
docker.login() {
    local i
    local parameters

    for i in $(cat ${HOME}/tokens/docker)
    do
        parameters+=" --${i}"
    done
    docker login ${parameters} ${HUB_ADDRESS}
}

#Example of usage: get_image_name ${name} ${version} ${tag} ${engine} ${repository_type} ${frontend}
docker.get_image_name() {
    local name="${1}"
    local version="${2}"
    local tag="${3}"
    local engine="${4}"
    local repository_type="${5}"
    local frontend="${6}"

    if [ "_${repository_type,,}" == "_private" ]; then
        repository_type="_${repository_type}"
    else
        repository_type=""
    fi

    if [ ! -z ${engine} ]; then
        engine="_${engine}"
    fi

    if [ ! -z ${frontend} ]; then
        frontend="_${frontend}"
    fi

    echo "${HUB_ADDRESS}/aibt_${name,,}${repository_type,,}/${version,,}${engine,,}${frontend,,}:${tag}"
}

docker.get_git_token() {
    local token=$(cat ${HOME}/tokens/private_git)
    echo "${token}"
}

#Example of usage: build ${image_name} ${dockerfile}
docker.build() {
    local image_name="${1}"
    local dockerfile_path="${2}"
    local repository_type="${3}"
    local build_cores_number="${4}"

    if [ ${repository_type} == "private" ]; then
        BUILD_ARGS="--build-arg REPOSITORY_TYPE=private --build-arg TOKEN=$(docker.get_git_token)"
    fi

    # Add http_proxy if exists
    if [ -n ${http_proxy} ]; then
        BUILD_ARGS+="--build-arg http_proxy=${http_proxy} "
    fi

    # Add https_proxy if exists
    if [ -n ${https_proxy} ]; then
        BUILD_ARGS+="--build-arg https_proxy=${https_proxy} "
    fi

    # If build_cores_number was not passed - detect number of build cores
    if [ -z ${build_cores_number} ]; then
        BUILD_ARGS+="--build-arg BUILD_CORES_NUMBER=$(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l) "
    fi

    docker build ${BUILD_ARGS} -f "${WORKDIR}/${dockerfile_path}" -t "${image_name}" .
    local exit_code=${?}
    if [ ${exit_code} != "0" ]; then
        exit ${exit_code}
    fi
}

#Example of usage: push ${image_name}
docker.push() {
    local image_name="${1}"

    docker.login
    docker push "${image_name}"
}

#Example of usage: pull ${image_name}
docker.pull() {
    local image_name="${1}"

    docker.login
    docker pull "${image_name}"
}

#Example of usage: shell ${image_name} ${container_name} ${volumes} ${env} ${ports}
docker.shell() {
    local image_name=${1}
    local container_name="${2}"
    local volumes="${3}"
    local env="${4}"
    local ports="${5}"

    docker run -h "$(hostname)" --rm --privileged --name "${container_name}" -i -t "${ports}" "${volumes}" "${env}" \
        "${image_name}" /bin/bash
}

#Example of usage: run ${image_name} ${container_name} ${volumes} ${env} ${ports}
docker.run() {
    local image_name="${1}"
    local container_name="${2}"
    local volumes="${3}"
    local env="${4}"
    local ports="${5}"
    local engine="${6}"

    DOCKER_COMMAND="docker"

    if [ ${engine,,} == "cudnn" ]; then
        DOCKER_COMMAND="nvidia-docker"
    fi

    ${DOCKER_COMMAND} run -h "$(hostname)" --rm --privileged --name "${container_name}" "${ports}" "${volumes}" "${env}" "${image_name}"
}

#Example of usage: start ${image_name} ${container_name} ${volumes} ${env} ${ports}
docker.start() {
    local image_name="${1}"
    local container_name="${2}"
    local volumes="${3}"
    local env="${4}"
    local ports="${5}"
    local engine="${6}"

    docker ps -a | grep "${container_name}"  &> /dev/null
    if [ $? == 0 ]; then
        docker.stop "${container_name}"
        docker.remove "${container_name}"
    fi

    DOCKER_COMMAND="docker"

    if [ ${engine,,} == "cudnn" ]; then
        DOCKER_COMMAND="nvidia-docker"
    fi

    CMD="${DOCKER_COMMAND} run -h "$(hostname)" -id --privileged --name "${container_name}" "${ports}" "${volumes}" "${env}" "${image_name}" tail -f /dev/null"

    eval "${CMD}"
}

#Example of usage: commit ${image_name} ${container_name}
docker.commit() {
    local image_name="${1}"
    local container_name="${2}"

    docker commit "${container_name}" "${image_name}"
}

#Example of usage: tag ${image_name} ${new_tag}
docker.tag() {
    local image_name="${1}"
    local new_tag="${2}"

    docker tag "${image_name}" "${image_name/:*/:${new_tag}}"
}

#Example of usage: release ${image_name}
docker.release() {
    local image_name="${1}"

    docker.tag "${image_name}" "latest"
}

#Example of usage: stop ${container_name}
docker.stop() {
    local container_name="${1}"

    docker stop "${container_name}" || true
}

#Example of usage: clean_workdir ${container_name} ${directory} ${options}
docker.chmod() {
    local container_name="${1}"
    local directory="${2}"
    local options="${3}"

    docker start ${container_name}
    docker exec ${container_name} bash -c "cd ${directory}; chmod ${options} \$(ls -a | tail -n +3)"
}

#Example of usage: remove ${container_name}
docker.remove() {
    local container_name="${1}"

    docker rm "${container_name}" || true
}

#Example of usage: prune
docker.format() {
    docker system prune --all --force
}

#Example of usage: clean_up
docker.clean_up() {
    #list of all container
    local -r containers_list="$(docker ps -a -q)"
    #list of wrongly taged images
    local -r images_list="$(docker images --format "{{.Repository}}:{{.Tag}}->{{.ID}}" | grep '<none>')"

    if [[ ! -z "${containers_list}" ]]; then
        #Stop containers
        docker stop ${containers_list}
        #Remove containers
        docker rm ${containers_list}
    fi
    if [[ ! -z "${images_list}" ]]; then
        # Delete images
        for image in ${images_list}
        do
            docker rmi ${image/->*/} #remove by name
            docker rmi ${image/*->/} #remove by id
        done
    fi
    #Clean docker system
    printf 'y' | docker system prune
}

#Script help
usage() {
    cat <<EOF
    Usage: $0 [options]
EOF
}

main() {
    local pattern='[-a-zA-Z0-9_]*='
    local i
    local action=${1}; shift #assign first argument and remove from the argument list

    #parse arguments
    for i in "${@}"
    do
        local parameter_name
        for parameter_name in "${PARAMETERS[@]}"
        do
            if [[ ${i} == "--${parameter_name}="* ]]; then
                local value="${i//${pattern}/}"
                eval "local ${parameter_name}=\"${value}\""
            fi
        done
    done
    if [ -z ${image_name} ]; then
        local image_name="$(docker.get_image_name ${name} ${version} ${tag:-"latest"} ${engine:-"base"} ${repository_type:-"public"} ${frontend})"
    fi
    case "${action}" in
        build)
            docker.build "${image_name}" "${dockerfile_path}" "${repository_type:-"public"}" "${build_cores_number}";;
        push)
            docker.push "${image_name}";;
        pull)
            docker.pull "${image_name}";;
        shell)
            docker.shell "${image_name}" "${container_name}" "${volumes}" "${env}" "${ports}";;
        run)
            docker.run "${image_name}" "${container_name}" "${volumes}" "${env}" "${ports}" "${engine:-"base"}";;
        start)
            docker.start "${image_name}" "${container_name}" "${volumes}" "${env}" "${ports}" "${engine:-"base"}";;
        commit)
            docker.commit "${image_name}" "${container_name}";;
        tag)
            docker.tag "${image_name}" "${new_tag}";;
        stop)
            docker.stop "${container_name}";;
        remove)
            docker.remove "${container_name}";;
        chmod)
            docker.chmod "${container_name}" "${directory}" "${options}";;
        format)
            docker.format;;
        clean_up)
            docker.clean_up;;
        login)
            docker.login;;
        release)
            docker.release "${image_name}";;
        *)
            usage;;
    esac
}

if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    main "${@}"
fi
