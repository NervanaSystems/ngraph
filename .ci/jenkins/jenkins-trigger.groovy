// This script acts as a trigger script for the main ngraph-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

// The original ngraph-unittest job required the following parameters.  We
// set these up below as global variables, so we do not need to rewrite the
// original script -- we only need to provide this new trigger hook.
//
// ngraph-unittest parameters:
PR_URL = CHANGE_URL
PR_COMMIT_AUTHOR = CHANGE_AUTHOR
JENKINS_BRANCH = "chrisl/new-ci-trigger"
TIMEOUTTIME = "3600"
// BRANCH parameter is no loner needed
// TRIGGER_URL parameter is no longer needed

// Constants
JENKINS_DIR="."

env.MB_PIPELINE_CHECKOUT = true

node("bdw && nogpu") {

    deleteDir()  // Clear the workspace before starting

    echo "jenkins-trigger parameters:"
    echo "PR_URL           = ${PR_URL}"
    echo "PR_COMMIT_AUTHOR = ${PR_COMMIT_AUTHOR}"
    echo "JENKINS_BRANCH   = ${JENKINS_BRANCH}"
    echo "TIMEOUTTIME      = ${TIMEOUTTIME}"

    // Clone the cje-algo directory which contains our Jenkins groovy scripts
    git(branch: JENKINS_BRANCH, changelog: false, poll: false,
        url: 'https://github.intel.com/AIPG/cje-algo')

    // Call the main job script.
    //
    // NOTE: We keep the main job script in github.intel.com because it may
    //      contain references to technology which has not yet been released.
    //
    echo "Calling ngraph-ci-premerge.groovy"
    def ngraphCIPreMerge = load("${JENKINS_DIR}/ngraph-ci-premerge.groovy")
    ngraphCIPreMerge()
    echo "ngraph-ci-premerge.groovy completed"

}  // End:  node( ... )

echo "Done"

