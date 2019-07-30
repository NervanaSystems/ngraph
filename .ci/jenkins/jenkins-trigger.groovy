// Copyright 2017-2019 Intel Corporation
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

// This script acts as a trigger script for the main ngraph-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

// The original ngraph-unittest job required the following parameters.  We
// set these up below as global variables, so we do not need to rewrite the
// original script -- we only need to provide this new trigger hook.
//
// Parameters which ngraph-unittest uses:
String  PR_URL = CHANGE_URL
String  PR_COMMIT_AUTHOR = CHANGE_AUTHOR
String  PR_TARGET = CHANGE_TARGET
String  JENKINS_BRANCH = "master"
Integer TIMEOUTTIME = "3600"
// BRANCH parameter is no loner needed
// TRIGGER_URL parameter is no longer needed

// Constants
JENKINS_DIR = '.'

env.MB_PIPELINE_CHECKOUT = true

timestamps {
    node("trigger") {

        deleteDir()  // Clear the workspace before starting

        // Clone the cje-algo directory which contains our Jenkins groovy scripts
        try {
            sh "git clone -b $JENKINS_BRANCH https://github.intel.com/AIPG/cje-algo ."
        } catch (e) {
            echo "${e}"
            println("ERROR: An error occurred during cje-algo script checkout.")
            throw e
        }

        // Call the main job script.
        //
        // NOTE: We keep the main job script in github.intel.com because it may
        //      contain references to technology which has not yet been released.
        //
        
        echo "Calling ngraph-ci-premerge.groovy"
        def ngraphCIPreMerge = load("${JENKINS_DIR}/ngraph-ci-premerge.groovy")
        ngraphCIPreMerge(PR_URL, PR_COMMIT_AUTHOR, JENKINS_BRANCH, TIMEOUTTIME, PR_TARGET)
        echo "ngraph-ci-premerge.groovy completed"

    }  // End:  node
}  // End:  timestamps

echo "Done"

