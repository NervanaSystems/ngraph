// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation All Rights Reserved.
// The source code contained or described herein and all documents related to the
// source code ("Material") are owned by Intel Corporation or its suppliers or
// licensors. Title to the Material remains with Intel Corporation or its
// suppliers and licensors. The Material may contain trade secrets and proprietary
// and confidential information of Intel Corporation and its suppliers and
// licensors, and is protected by worldwide copyright and trade secret laws and
// treaty provisions. No part of the Material may be used, copied, reproduced,
// modified, published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery of
// the Materials, either expressly, by implication, inducement, estoppel or
// otherwise. Any license under such intellectual property rights must be express
// and approved by Intel in writing.

// CI settings
PROJECT_NAME = "ngraph"
WORKDIR = "${WORKSPACE}/${BUILD_NUMBER}/${PROJECT_NAME}"
CI_ROOT = ".ci/jenkins"
DOCKER_CONTAINER_NAME = "jenkins_${PROJECT_NAME}_ci"

UTILS = load "${CI_ROOT}/utils/utils.groovy"
result = 'SUCCESS'


def BuildImage(configurationMaps) {
    Closure buildMethod = { configMap ->
        sh """
            ${CI_ROOT}/utils/docker.sh build \
                                --name=${configMap["projectName"]} \
                                --version=${configMap["name"]} \
                                --dockerfile_path=${configMap["dockerfilePath"]}
        """
    }
    UTILS.CreateStage("Build_Image", buildMethod, configurationMaps)
}

def RunDockerContainers(configurationMaps) {
    Closure runContainerMethod = { configMap ->
        UTILS.PropagateStatus("Build_Image", configMap["name"])
        sh """
            mkdir -p ${HOME}/NGRAPH_CI
            ${CI_ROOT}/utils/docker.sh start \
                                --name=${configMap["projectName"]} \
                                --version=${configMap["name"]} \
                                --container_name=${configMap["dockerContainerName"]} \
                                --volumes="-v ${WORKSPACE}/${BUILD_NUMBER}:/logs -v ${WORKDIR}:/root"
        """
    }
    UTILS.CreateStage("Run_docker_containers", runContainerMethod, configurationMaps)
}

def BuildNgraph(configurationMaps) {
    Closure buildNgraphMethod = { configMap ->
        UTILS.PropagateStatus("Run_docker_containers", configMap["dockerContainerName"])
        sh """
            docker exec ${configMap["dockerContainerName"]} ./root/${CI_ROOT}/build_ngraph.sh
        """
    }
    UTILS.CreateStage("Build_NGraph", buildNgraphMethod, configurationMaps)
}

def RunToxTests(configurationMaps) {
    Closure runToxTestsMethod = { configMap ->
        UTILS.PropagateStatus("Build_NGraph", configMap["dockerContainerName"])
        sh """
            NGRAPH_WHL=\$(docker exec ${configMap["dockerContainerName"]} find /root/ngraph/python/dist/ -name 'ngraph*.whl')
            docker exec -e TOX_INSTALL_NGRAPH_FROM=\${NGRAPH_WHL} ${configMap["dockerContainerName"]} tox -c /root
        """
    }
    UTILS.CreateStage("Run_tox_tests", runToxTestsMethod, configurationMaps)
}

def Cleanup(configurationMaps) {
    Closure cleanupMethod = { configMap ->
        sh """
            ${CI_ROOT}/utils/docker.sh chmod --container_name=${configMap["dockerContainerName"]} --directory="/logs" --options="-R 777" || true
            ${CI_ROOT}/utils/docker.sh stop --container_name=${configMap["dockerContainerName"]} || true
            ${CI_ROOT}/utils/docker.sh remove --container_name=${configMap["dockerContainerName"]} || true
            ${CI_ROOT}/utils/docker.sh clean_up || true
        """
    }
    UTILS.CreateStage("Cleanup", cleanupMethod, configurationMaps)
}

def Notify() {
    configurationMaps = []
    configurationMaps.add([
        "name": "notify"
    ])
    Closure notifyMethod = { configMap ->
        UTILS.PropagateStatus("Cleanup", configMap["dockerContainerName"])
        if(currentBuild.result != "FAILURE") {
            currentBuild.result = "SUCCESS"
        }
        emailext (
            subject: "NGraph-Onnx CI: NGraph PR $ghprbPullId",
            body: """
                <table style="width:100%">
                    <tr><td>Status:</td> <td>${currentBuild.result}</td></tr>
                    <tr><td>Repository</td> <td>$ghprbGhRepository</td></tr>
                    <tr><td>Branch:</td> <td>$ghprbSourceBranch</td></tr>
                    <tr><td>Pull Request:</td> <td>$ghprbPullId</td></tr>
                    <tr><td>Commit SHA:</td> <td>$ghprbActualCommit</td></tr>
                    <tr><td>Link:</td> <td>$ghprbPullLink</td></tr>
                </table>
            """,
            to: "$ghprbActualCommitAuthorEmail"
        )
    }
    UTILS.CreateStage("Notify", notifyMethod, configurationMaps)
}

def main(String projectName, String projectRoot, String dockerContainerName) {
    timeout(activity: true, time: 60) {
        def configurationMaps = UTILS.GetDockerEnvList(projectName, dockerContainerName, projectRoot)
        BuildImage(configurationMaps)
        RunDockerContainers(configurationMaps)
        BuildNgraph(configurationMaps)
        RunToxTests(configurationMaps)
        Cleanup(configurationMaps)
        Notify()
    }
}

main(PROJECT_NAME, CI_ROOT, DOCKER_CONTAINER_NAME)