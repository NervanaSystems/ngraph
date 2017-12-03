# Basic Makefile for contrib/docker. This can be expanded later as more targets
# are added.

# Default is to build with -j for parallel builds.  Turn off with
#   make PARELLEL=
PARALLEL=-j

# DIR is an internal variable that serves as an anchor to this cloned git
# repository.  DIR is mounted into the docker container, so that builds
# can occur within the container on this cloned git repository.  DIR should
# not be modified - if it is, then the build system will not work.
DIR = $(realpath ../..)

# DOCKUSER_HOME is the location of the home directory of the fabricated
# "dockuser" user, used only within the docker containers.  "dockuser" is
# created (from the passed-in RUN_UID) to map the docker-caller user's UID to a
# first-class user (/etc/passwd entry, member of sudo group, proper home dir)
# /home/dockuser is also used in other scripts, notably run_as_user.sh, so if
# changed it must be done in other areas for the builds to work.
DOCKUSER_HOME=/home/dockuser

# Use /home/dockuser/ngraph-cpp-test, because we run as the user (and not root)
# /root/ngraph-cpp-test is not used, because /root is not accessible to user
VOLUME = -v "${DIR}:${DOCKUSER_HOME}/ngraph-cpp-test"
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
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

expand_dockerfile_templates:
	cd "${DIR}"/contrib/docker
	mkdir "${BUILD_DIR}" || true
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.ngraph_cpp_cpu > "${BUILD_DIR}"/Dockerfile.ngraph_cpp_cpu

clean:
	rm -f "${DIR}"/contrib/docker/.build-*/Dockerfile.* || echo "keep going if files are not present"
	rmdir "${DIR}"/contrib/docker/.build-* || echo "keep going if directory is not present"
	rm -fr "${DIR}"/BUILD

build_ngraph_cpp_cpu: expand_dockerfile_templates
	$(DOCKER_BUILD) -f="${BUILD_DIR}"/Dockerfile.ngraph_cpp_cpu --build-arg python_version="${PYTHON_VERSION}" -t=ngraph_cpp_cpu:"${BUILD_VERSION}" "${DIR}"
	# remove the tag for the previous latest image
	docker rmi ngraph_cpp_cpu:latest || echo "keep going if docker rmi command fails"
	docker tag `docker images -q "ngraph_cpp_cpu:${BUILD_VERSION}"` ngraph_cpp_cpu:latest

build_all: build_ngraph_cpp_cpu

check_cpu: build_ngraph_cpp_cpu
	# Remove old distribution directory if present
	( test -d "${DIR}"/BUILD/ngraph_dist && rm -fr "${DIR}"/BUILD/ngraph_dist && echo "Removed old ${DIR}/BUILD/ngraph_dist directory" ) || echo "Previous ngraph_dist directory not found"
	# Make BUILD directory as user
	mkdir -p "${DIR}"/BUILD
	chmod ug+rwx "${DIR}"/BUILD
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env GTEST_OUTPUT="xml:${DOCKUSER_HOME}/ngraph-cpp-test/BUILD/unit-test-results.xml" \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="set -e ; set -o pipefail ; cd ${DOCKUSER_HOME}/ngraph-cpp-test/BUILD; cmake -DCMAKE_CXX_COMPILER=clang++-3.9 -DCMAKE_C_COMPILER=clang-3.9 -DNGRAPH_BUILD_DOXYGEN_DOCS=ON -DNGRAPH_BUILD_SPHINX_DOCS=ON -DNGRAPH_WARNINGS_AS_ERRORS=ON .. 2>&1 | tee cmake.log ; env VERBOSE=1 make ${PARALLEL} 2>&1 | tee make.log ; env VERBOSE=1 make check 2>&1 | tee make_check.log" \
            "ngraph_cpp_cpu:${BUILD_VERSION}" \
	    sh -c "${DOCKUSER_HOME}/ngraph-cpp-test/contrib/docker/run_as_user.sh"

shell: build_ngraph_cpp_cpu
	# "make shell" runs an interactive shell in the docker image, for debugging
	docker run --rm --tty --interactive \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID="$(shell id -u)" \
            "ngraph_cpp_cpu:${BUILD_VERSION}" \
            sh -c "cd ${DOCKUSER_HOME} ; ${DOCKUSER_HOME}/ngraph-cpp-test/contrib/docker/run_as_user.sh"

install:
	# Puts ngraph_dist in BUILD directory.  This is used by Jenkins ngraph-tensorflow batch job.
	# Note: We currently have a bug where cmake only installs in $HOME.  Jira NGTF-205 is opened
	#       for this.  For now, here we install to $HOME, then move the directory.
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="set -e ; set -o pipefail; cd ${DOCKUSER_HOME}/ngraph-cpp-test/BUILD ; test -d ngraph_dist && rm -fr ngraph_dist && echo 'Removed old ngraph_dist directory' ; make install 2>&1 | tee make_install.log ; mv -v ${DOCKUSER_HOME}/ngraph_dist ${DOCKUSER_HOME}/ngraph-cpp-test/BUILD" \
            "ngraph_cpp_cpu:${BUILD_VERSION}" \
	    sh -c "${DOCKUSER_HOME}/ngraph-cpp-test/contrib/docker/run_as_user.sh"

all: build_ngraph_cpp_cpu
