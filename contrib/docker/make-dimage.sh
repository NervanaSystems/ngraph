#!  /bin/bash

###
#
# Create a docker image that includes dependencies for building ngraph
#
# Uses CONTEXTDIR as the docker build context directory
#   Default value is '.'
#
# Uses ./Dockerfile.${DOCKER_TAG}
#   DOCKER_TAG is set to 'ngraph' if not set 
#
# Sets the docker image name as ${DOCKER_IMAGE_NAME}
#   DOCKER_IMAGE_NAME is set to the ${DOCKER_TAG} if not set in the environment
#   The datestamp tag is automatically appended to the DOCKER_IMAGE_NAME to create the DIMAGE_ID
#   The ${DIMAGE_ID} docker image is created on the local server
#   The ${DOCKER_IMAGE_NAME}:latest tag is also created by default for reference
#
###

set -e
#set -u
set -o pipefail

if [ -z $DOCKER_TAG ]; then
    DOCKER_TAG=build_ngraph
fi

if [ -z $DOCKER_IMAGE_NAME ]; then
    DOCKER_IMAGE_NAME=${DOCKER_TAG}
fi

echo "CONTEXTDIR=${CONTEXTDIR}"

if [ -z ${CONTEXTDIR} ]; then
    CONTEXTDIR='.'  # Docker image build context
fi

echo "CONTEXTDIR=${CONTEXTDIR}"

if [ -n $DFILE ]; then
    DFILE="${CONTEXTDIR}/Dockerfile.${DOCKER_TAG}"
fi

CONTEXT='.'

DIMAGE_NAME="${DOCKER_IMAGE_NAME}"
DIMAGE_VERSION=`date -Iseconds | sed -e 's/:/-/g'`

DIMAGE_ID="${DIMAGE_NAME}:${DIMAGE_VERSION}"

cd ${CONTEXTDIR}

echo ' '
echo "Building docker image ${DIMAGE_ID} from Dockerfile ${DFILE}, context ${CONTEXT}"
echo ' '

# build the docker base image
docker build  --rm=true \
       -f="${DFILE}" \
       --build-arg http_proxy=http://proxy-us.intel.com:911 \
       --build-arg https_proxy=https://proxy-us.intel.com:911 \
       -t="${DIMAGE_ID}" \
       ${CONTEXT}

docker tag  "${DIMAGE_ID}"  "${DIMAGE_NAME}:latest"

echo ' '
echo 'Docker image build completed'
echo ' '
