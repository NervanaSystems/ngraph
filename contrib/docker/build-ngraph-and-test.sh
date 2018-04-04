#!  /bin/bash

set -e
# set -u  # Cannot use set -u, as activate below relies on unbound variables
set -o pipefail

# Debugging to verify builds on CentOS 7.4 and Ubuntu 16.04
if [ -f "/etc/centos-release" ]; then
    cat /etc/centos-release
fi

if [ -f "/etc/lsb-release" ]; then
    cat /etc/lsb-release
fi

uname -a
cat /etc/os-release || true

echo ' '
echo 'Contents of /home:'
ls -la /home
echo ' '
echo 'Contents of /home/dockuser:'
ls -la /home/dockuser
echo ' '

export CMAKE_OPTIONS_EXTRA=""

# setting for make -j
if [ -z ${PARALLEL} ] ; then
    PARALLEL=22
fi

# make command to execute
if [ -z ${CMD_TO_RUN} ] ; then
    CMD_TO_RUN='check_gcc'
fi

# directory name to use for the build
if [ -z ${BUILD_SUBDIR} ] ; then
    BUILD_SUBDIR=BUILD
fi

# Option to build with GPU backend
# default builds with CPU only
if [ -z ${NGRAPH_GPU_ENABLE} ] ; then
    NGRAPH_GPU_ENABLE=false
fi

# Set up the environment
if $NGRAPH_GPU_ENABLE; then
    export CMAKE_OPTIONS_EXTRA="-DNGRAPH_GPU_ENABLE=TRUE"
fi

export NGRAPH_REPO=/home/dockuser/ngraph-test

if [ -z ${OUTPUT_DIR} ]; then
    OUTPUT_DIR="${NGRAPH_REPO}/${BUILD_SUBDIR}"
fi

# Remove old OUTPUT_DIR directory if present for build_* targets
if [ "$(echo ${CMD_TO_RUN} | grep build | wc -l)" != "0" ] ; then
    ( test -d ${OUTPUT_DIR} && rm -fr ${OUTPUT_DIR} && echo "Removed old ${OUTPUT_DIR} directory" ) || echo "Previous ${OUTPUT_DIR} directory not found"
    # Make OUTPUT_DIR directory as user
    mkdir -p ${OUTPUT_DIR}
    chmod ug+rwx ${OUTPUT_DIR}

    # Use a third-party cache if it is present
    # If THIRD_PARTY_CACHE_DIR is set in the environment,
    #    if the directory exists,
    #        set up a symbolic link to the 'third-party' directory from the OUTPUT_DIR
    #        so the cmake/make process will not download and rebuild the third-party dependencies
    #        if they match the criteria set in the cmake files
    #    else
    #        record a message to the log that although the variable was set, the directory was not
    #        found in the build environment
    #
    # TODO: This is a workaround during testing of the EXTERNAL_PROJECTS_ROOT cmake setting
    #
    if [ -z ${THIRD_PARTY_CACHE_DIR} ]; then
        echo "No THIRD_PARTY_CACHE_DIR specified - will download and build third-party dependencies"
    else
        if [ -d ${THIRD_PARTY_CACHE_DIR} ]; then
            cd ${OUTPUT_DIR}
            ln -s "${THIRD_PARTY_CACHE_DIR}/third-party" "third-party"
        else
            echo "THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} defined, but not found -"
            echo "will download and build third-party dependencies"
        fi
    fi
fi

#TODO: add openmpi dependency to enable building with -DNGRAPH_DISTRIBUTED_ENABLE=TRUE
#
#if [ -z ${NGRAPH_DISTRIBUTED_ENABLE} ] ; then
#  NGRAPH_DISTRIBUTED_ENABLE=false
#fi

#if $NGRAPH_DISTRIBUTED_ENABLE; then
#   source /home/environment-openmpi-ci.source
#   which mpirun
#   mpirun --version
#   export CMAKE_OPTIONS_EXTRA="-DNGRAPH_DISTRIBUTED_ENABLE=$NGRAPH_DISTRIBUTED_ENABLE"
#fi

GCC_VERSION=` gcc --version | grep gcc | cut -f 2 -d ')' | cut -f 2 -d ' ' | cut -f 1,2 -d '.'`

if [ "${GCC_VERSION}" != "4.8" ] ; then
    export CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA} -DNGRAPH_USE_PREBUILT_LLVM=TRUE"
fi

# Print the environment, for debugging
echo ' '
echo 'Environment:'
export
echo ' '

cd $NGRAPH_REPO

export CMAKE_OPTIONS_COMMON="-DNGRAPH_BUILD_DOXYGEN_DOCS=ON -DNGRAPH_BUILD_SPHINX_DOCS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ${CMAKE_OPTIONS_EXTRA}"
export CMAKE_OPTIONS_GCC="${CMAKE_OPTIONS_COMMON} -DNGRAPH_INSTALL_PREFIX=${NGRAPH_REPO}/BUILD-GCC/ngraph_dist"
export CMAKE_OPTIONS_CLANG="$CMAKE_OPTIONS_COMMON -DNGRAPH_INSTALL_PREFIX=${NGRAPH_REPO}/BUILD-CLANG/ngraph_dist -DCMAKE_CXX_COMPILER=clang++-3.9 -DCMAKE_C_COMPILER=clang-3.9 -DNGRAPH_WARNINGS_AS_ERRORS=ON -DNGRAPH_USE_PREBUILT_LLVM=TRUE"

echo "CMD_TO_RUN=${CMD_TO_RUN}"

# set up the cmake environment
if [ -z ${CMAKE_OPTIONS} ] ; then
    if [ "$(echo ${CMD_TO_RUN} | grep gcc | wc -l)" != "0" ] ; then
        export CMAKE_OPTIONS=${CMAKE_OPTIONS_GCC}
    elif [ "$(echo ${CMD_TO_RUN} | grep clang | wc -l)" != "0" ] ; then
        export CMAKE_OPTIONS=${CMAKE_OPTIONS_CLANG}
    else
        export CMAKE_OPTIONS=${CMAKE_OPTIONS_COMMON}
    fi

    echo "set CMAKE_OPTIONS=${CMAKE_OPTIONS}"
fi

# build and test
export BUILD_DIR="${NGRAPH_REPO}/${BUILD_SUBDIR}"
export GTEST_OUTPUT="xml:${BUILD_DIR}/unit-test-results.xml"
mkdir -p ${BUILD_DIR}
chmod ug+rwx ${BUILD_DIR}
cd ${BUILD_DIR}

echo "Build and test for ${CMD_TO_RUN} in `pwd` with specific parameters:"
echo "    NGRAPH_REPO=${NGRAPH_REPO}"
echo "    CMAKE_OPTIONS=${CMAKE_OPTIONS}"
echo "    GTEST_OUTPUT=${GTEST_OUTPUT}"

# only run cmake/make steps for build_* make targets
if [ "$(echo ${CMD_TO_RUN} | grep build | wc -l)" != "0" ] ; then
    # always run cmake/make steps
    echo "Running cmake"
    cmake ${CMAKE_OPTIONS} .. 2>&1 | tee ${OUTPUT_DIR}/cmake_${CMD_TO_RUN}.log
    echo "Running make"
    env VERBOSE=1 make -j ${PARALLEL} 2>&1 | tee ${OUTPUT_DIR}/make_${CMD_TO_RUN}.log
    echo "CMD_TO_RUN=${CMD_TO_RUN} finished - cmake/make steps completed"
else
    # strip off _* from CMD_TO_RUN to pass to the ngraph make targets 
    MAKE_CMD_TO_RUN=`echo ${CMD_TO_RUN} | sed 's/_.*//g'`
    COMPILER=`echo ${CMD_TO_RUN} | sed 's/.*_//g'`

    if [ "${MAKE_CMD_TO_RUN}" == "unit-test-check" ]; then
    # check style before running unit tests
        if [ -f "/usr/bin/clang-3.9" ]; then
            echo "Running make style-check"
            env VERBOSE=1 make -j style-check 2>&1 | tee ${OUTPUT_DIR}/make_style_check_${CMD_TO_RUN}.log
        fi
    fi

    echo "Running make ${MAKE_CMD_TO_RUN}"
    env VERBOSE=1 make ${MAKE_CMD_TO_RUN} 2>&1 | tee ${OUTPUT_DIR}/make_${CMD_TO_RUN}.log

    if [ "${MAKE_CMD_TO_RUN}" == "install" ] ; then
        echo "Building ngraph_dist_${COMPILER}.tgz"
        tar czf ngraph_dist_${COMPILER}.tgz ngraph_dist 2>&1 | tee make_tarball_${COMPILER}.log
        ls -l ngraph_dist_*.tgz
    fi
fi

