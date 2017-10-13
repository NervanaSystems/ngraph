# Basic Makefile for contrib/docker. This can be expanded later as more targets
# are added.

# Default is to build with -j for parallel builds.  Turn off with
#   make PARELLEL=
PARALLEL=-j

DIR = $(realpath ../..)
# Use /home/dockuser/ngraph-cpp-test, because we run as the user (and not root)
# /root/ngraph-cpp-test is not used, because /root is not accessible to user
VOLUME = -v ${DIR}:/home/dockuser/ngraph-cpp-test
GIT_COMMIT = $(shell git rev-parse HEAD)
BUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}
BUILD_DIR = ${DIR}/contrib/docker/.build-${BUILD_VERSION}

CALLER_UID := $(shell id -u)
CALLER_GID := $(shell id -g)

# Default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

.PHONY: clean build_ngraph_cpp_cpu check_cpu install shell build_all

DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env https_proxy=$(https_proxy)
endif

expand_dockerfile_templates:
	cd ${DIR}/contrib/docker
	mkdir ${BUILD_DIR} || true
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.ngraph_cpp_cpu > ${BUILD_DIR}/Dockerfile.ngraph_cpp_cpu

clean:
	rm -f ${DIR}/contrib/docker/.build-*/Dockerfile.* || echo "keep going if files are not present"
	rmdir ${DIR}/contrib/docker/.build-* || echo "keep going if directory is not present"
	rm -fr ${DIR}/BUILD

build_ngraph_cpp_cpu: expand_dockerfile_templates
	$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.ngraph_cpp_cpu --build-arg python_version=${PYTHON_VERSION} -t=ngraph_cpp_cpu:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	docker rmi ngraph_cpp_cpu:latest || echo "keep going if docker rmi command fails"
	docker tag `docker images -q ngraph_cpp_cpu:${BUILD_VERSION}` ngraph_cpp_cpu:latest

build_all: build_ngraph_cpp_cpu

check_cpu: build_ngraph_cpp_cpu
	# Remove old distribution directory if present
	( test -d ${DIR}/BUILD/ngraph_dist && rm -fr ${DIR}/BUILD/ngraph_dist && echo "Removed old ${DIR}/BUILD/ngraph_dist directory" ) || echo "Previous ngraph_dist directory not found"
	# Make BUILD directory as user
	mkdir -p ${DIR}/BUILD
	chmod ug+rwx ${DIR}/BUILD
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID=`id -u` \
            --env RUN_CMD='set -e ; set -o pipefail ; cd /home/dockuser/ngraph-cpp-test/BUILD; cmake -DCMAKE_CXX_COMPILER=clang++-3.9 -DCMAKE_C_COMPILER=clang-3.9 .. 2>&1 | tee cmake.log ; env VERBOSE=1 make ${PARALLEL} 2>&1 | tee make.log ; env VERBOSE=1 make check 2>&1 | tee make_check.log' \
            ngraph_cpp_cpu:${BUILD_VERSION} \
	    sh -c '/home/dockuser/ngraph-cpp-test/contrib/docker/run_as_user.sh'

shell: build_ngraph_cpp_cpu
	docker run --rm --tty --interactive \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID=`id -u` \
            ngraph_cpp_cpu:${BUILD_VERSION} \
            sh -c 'cd /home/dockuser ; /home/dockuser/ngraph-cpp-test/contrib/docker/run_as_user.sh'

install:
	# Puts ngraph_dist in BUILD directory.  This is used by Jenkins ngraph-tensorflow batch job.
	# Note: We currently have a bug where cmake only installs in $HOME.  Jira NGTF-205 is opened
	#       for this.  For now, here we install to $HOME, then move the directory.
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID=`id -u` \
            --env RUN_CMD='set -e ; set -o pipefail; cd /home/dockuser/ngraph-cpp-test/BUILD ; test -d ngraph_dist && rm -fr ngraph_dist && echo "Removed old ngraph_dist directory" ; make install 2>&1 | tee make_install.log ; mv -v /home/dockuser/ngraph_dist /home/dockuser/ngraph-cpp-test/BUILD' \
            ngraph_cpp_cpu:${BUILD_VERSION} \
	    sh -c '/home/dockuser/ngraph-cpp-test/contrib/docker/run_as_user.sh'

all: build_ngraph_cpp_cpu
