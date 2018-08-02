// This script acts as a trigger script for the main ngraph-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

node('bdw && nogpu') {

    // The original ngraph-unittest job required the following parameters.  We
    // set these up below as global variables, so we do not need to rewrite the
    // original script -- we only need to provide this new trigger hook.
    //
    // ngraph-unittest parameters:
    BRANCH = BRANCH_NAME
    // PR_URL = CHANGE_URL
    // PR_COMMIT_AUTHOR = CHANGE_AUTHOR
    // TRIGGER_URL        <- No longer needed
    JENKINS_BRANCH = "chrisl/new-ngraph-ci-trigger"
    TIMEOUTTIME = 3600

    echo "jenkins-trigger parameters:"
    echo "BRANCH           = ${BRANCH}"
    // echo "PR_URL           = ${PR_URL}"
    // echo "PR_COMMIT_AUTHOR = ${PR_COMMIT_AUTHOR}"
    echo "JENKINS_BRANCH   = ${JENKINS_BRANCH}"
    echo "TIMEOUTTIME      = ${TIMEOUTTIME}"

    // Clone the cje-algo directory which contains our Jenkins groovy scripts
    git poll: false, url: 'https://github.intel.com/AIPG/cje-algo'
    echo "After cloning cje-algo, workspace looks like:"
    sh "ls -l"

    // Call the main job script.
    //
    // NOTE: We keep the main job script in github.intel.com because it may
    //      contain references to technology which has not yet been released.
    //
    echo "Would call ngraph-unittest.groovy here"

}  // End:  node( ... )

echo "Done"

