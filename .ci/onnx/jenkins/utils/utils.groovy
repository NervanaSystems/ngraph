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
STAGES_STATUS_MAP = [:]

def GetDockerEnvList(String projectName, String dockerContainerNamePrefix, String projectRoot = projectName) {
    /**
    * This method generates configuration map list using dockerfiles available in dockerfiles directory
    *
    * @param projectName name of the project used in paths and configuration map.
    * @param dockerContainerNamePrefix docker container name prefix.
    * @param projectRoot path to project root containing directory with dockerfiles to run
    */

    def rawList = findFiles(glob: "${projectRoot}/dockerfiles/*.dockerfile")
    def envList = []
    for (int i = 0; i < rawList.size(); ++i) {
        def name = rawList[i].name - '.dockerfile'
        def dockerContainerName = "${dockerContainerNamePrefix}_${name}"
        envList.add([name:name, // name is the only obligatory vaiable
                     dockerfilePath:rawList[i].path,
                     projectName:projectName,
                     dockerContainerName:dockerContainerName])
    }
    return envList
}

def GenerateMap(Closure method, configurationMaps) {
    /**
    * Generates map for method using configurationMaps.
    *
    * @param method Method that will be executed in each map(configuration).
    * @param configurationMaps Map of configuration that will be parallelized.
    */

    def executionMap = [:]
    for (int i = 0; i < configurationMaps.size(); ++i) {
        configMap = configurationMaps[i]
        executionMap[configMap["name"]] = {
            method(configMap)
        }
    }
    return executionMap
}

def CreateStage(String stageName, Closure method, configurationMaps) {
    /**
    * Create pipeline stage.
    * 
    * @param stageName Name of stage that will be create.
    * @param method Method that will be executed in each map(configuration).
    * @param configurationMaps Map of configuration that will be parallelized.
    */

    stage(stageName) {
        // Add current stage name to configurationMaps
        for (int i = 0; i < configurationMaps.size(); ++i) {
            configurationMaps[i]["stageName"] = stageName
        }

        Closure genericBodyMethod = { configMap ->
            def status = "SUCCESS"
            try {
                method(configMap)
            } catch(e) {
                status = "FAILURE"
                throw e
            } finally {
                UTILS.SetConfigurationStatus(configMap["stageName"], configMap["name"], status)
            }
        }

        try {
            def prepareEnvMap = GenerateMap(genericBodyMethod, configurationMaps)
            parallel prepareEnvMap
        } catch(e) {
            Exception(e)
        }
    }
}

def SetConfigurationStatus(String stageName, String configurationName, String status) {
    /**
    * Set stage status.
    * 
    * @param stageName The name of the stage in which the configuration is.
    * @param configurationName The name of the configuration whose status will be updated.
    * @param status Configuration status: SUCCESS or FAILURE.
    */
    if (!STAGES_STATUS_MAP.containsKey(stageName)) {
        STAGES_STATUS_MAP[stageName] = [:]
    }
    if (["FAILURE", "SUCCESS"].contains(status.toUpperCase())) {
        STAGES_STATUS_MAP[stageName][configurationName] = status.toUpperCase()
    } else {
        throw new Exception("Not supported status name.")
    }
}

def PropagateStatus(String parentStageName, String parentConfigurationName) {
    /**
    * Popagate status in parent configuration fails.
    * This method will throw exeption "Propagating status of $parentStageName"
    * if parent configuration name status is FAILURE
    * 
    * @param parentStageName The name of the stage in which the configuration is.
    * @param parentConfigurationName The name of the configuration whose status will be propagated.
    */

    parentStageStatus = STAGES_STATUS_MAP[parentStageName][parentConfigurationName]
    if (parentStageStatus == "FAILURE") {
        throw new Exception("Propagating status of ${parentStageName}")
    }
}

def ShowStatusMap() {
    /**
    * Display status map for every defined stage.
    */

    echo "${STAGES_STATUS_MAP}"
}

def Exception(e) {
    currentBuild.result = 'FAILURE'
}

return this
