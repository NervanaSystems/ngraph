// ******************************************************************************
// Copyright 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ******************************************************************************
STAGES_STATUS_MAP = [:]

def getDockerEnvList(String projectName, String dockerContainerNamePrefix, String projectRoot = projectName) {
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

def generateMap(Closure method, configurationMaps) {
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

def createStage(String stageName, Closure method, configurationMaps, force = false) {
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

        // Fail current stage If earlier stage got aborted or failed
        // unless it's executed with force argument set to true
        Closure genericBodyMethod = {}
        if (!force && ["FAILURE", "ABORTED"].contains(currentBuild.result)) {
            genericBodyMethod = { configMap ->
                println("Skipping stage due to earlier stage ${currentBuild.result}")
                setConfigurationStatus(configMap["stageName"], configMap["name"], currentBuild.result)
                throw new Exception("Skipped due to ${currentBuild.result} in earlier stage")
            }
        }
        else
        {
            genericBodyMethod = { configMap ->
                def status = "SUCCESS"
                try {
                    method(configMap)
                } catch(Exception e) {
                    if (e.toString().contains("FlowInterruptedException")) {
                        status = "ABORTED"
                    } else {
                        status = "FAILURE"
                    }
                    currentBuild.result = status
                    throw e
                } finally {
                    setConfigurationStatus(configMap["stageName"], configMap["name"], status)
                }
            }
        }

        try {
            def prepareEnvMap = generateMap(genericBodyMethod, configurationMaps)
            parallel prepareEnvMap
        } catch(Exception e) {
            if (e.toString().contains("FlowInterruptedException")) {
                currentBuild.result = "ABORTED"
            } else {
                currentBuild.result = "FAILURE"
            }
        }
    }
}

def setConfigurationStatus(String stageName, String configurationName, String status) {
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
    if (["FAILURE", "SUCCESS", "ABORTED"].contains(status.toUpperCase())) {
        STAGES_STATUS_MAP[stageName][configurationName] = status.toUpperCase()
    } else {
        throw new Exception("Not supported status name.")
    }
}

def propagateStatus(String parentStageName, String parentConfigurationName) {
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

def showStatusMap() {
    /**
    * Display status map for every defined stage.
    */

    echo "${STAGES_STATUS_MAP}"
}

return this
