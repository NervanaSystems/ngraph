# Basic Makefile for contrib/docker. This can be expanded later as more targets
# are added.

DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/ngraph-cpp-test
GIT_COMMIT = $(shell git rev-parse HEAD)
BUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}
BUILD_DIR = ${DIR}/contrib/docker/.build-${BUILD_VERSION}

CALLER_UID := $(shell id -u)
CALLER_GID := $(shell id -g)

# Default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

.PHONY: clean build_ngraph_cpp_cpu check_cpu shell build_all

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

build_ngraph_cpp_cpu: expand_dockerfile_templates
	$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.ngraph_cpp_cpu --build-arg python_version=${PYTHON_VERSION} -t=ngraph_cpp_cpu:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	docker rmi ngraph_cpp_cpu:latest || echo "keep going if docker rmi command fails"
	docker tag `docker images -q ngraph_cpp_cpu:${BUILD_VERSION}` ngraph_cpp_cpu:latest

build_all: build_ngraph_cpp_cpu

check_cpu: build_ngraph_cpp_cpu
	docker run --rm ${DOCKER_RUN_ENV} ${VOLUME} -t ngraph_cpp_cpu:${BUILD_VERSION} sh -c "mkdir BUILD ; cd BUILD ; cmake .. ; make check"
	# update the files to be owned by the calling user instead of root, to avoid docker mount problems with file ownership
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-cpp-test \
	-t ngraph_cpp_cpu \
	/tmp/chown_files.sh

shell: build_ngraph_cpp_cpu
	docker run --rm ${VOLUME} -it ngraph_cpp_cpu:${BUILD_VERSION} /bin/bash

all: build_ngraph_cpp_cpu
