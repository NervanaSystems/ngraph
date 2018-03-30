#!  /bin/bash

set -e
# set -u  # Cannot use set -u, as activate below relies on unbound variables
set -o pipefail

# Debugging
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

if [ -z ${PARALLEL} ] ; then
    PARALLEL=22
fi

if [ -z ${BUILD_SUBDIR} ] ; then
    BUILD_SUBDIR=BUILD
fi

if [ -z ${TEST_SUITE} ] ; then
    TEST_SUITE=gcc
fi

if [ -z ${NGRAPH_GPU_ENABLE} ] ; then
    NGRAPH_GPU_ENABLE=false
fi

if [ -z ${RUN_UNIT_TESTS} ] ; then
    RUN_UNIT_TESTS=true
fi

# Set up the environment
if $NGRAPH_GPU_ENABLE; then
    export CMAKE_OPTIONS_EXTRA="-DNGRAPH_GPU_ENABLE=TRUE"
fi

export NGRAPH_REPO=/home/dockuser/ngraph-test

if [ -z ${OUTPUT_DIR} ]; then
    OUTPUT_DIR="${NGRAPH_REPO}/${BUILD_SUBDIR}"
fi

# Remove old OUTPUT_DIR directory if present
( test -d ${OUTPUT_DIR} && rm -fr ${OUTPUT_DIR} && echo "Removed old ${OUTPUT_DIR} directory" ) || echo "Previous ${OUTPUT_DIR} directory not found"
# Make OUTPUT_DIR directory as user
mkdir -p ${OUTPUT_DIR}
chmod ug+rwx ${OUTPUT_DIR}

#if [ -z ${NGRAPH_DISTRIBUTED_ENABLE} ] ; then
#  NGRAPH_DISTRIBUTED_ENABLE=false
#fi

#current problem with builds is that number of these options are not available in make files yet
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

echo "TEST_SUITE=${TEST_SUITE}"

if [ -z ${CMAKE_OPTIONS} ] ; then
    if [ "$(echo ${TEST_SUITE} | grep gcc | wc -l)" != "0" ] ; then
        export CMAKE_OPTIONS=${CMAKE_OPTIONS_GCC}
    elif [ "$(echo ${TEST_SUITE} | grep clang | wc -l)" != "0" ] ; then
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

echo "Build and test for ${TEST_SUITE} in `pwd` with specific parameters:"
echo "    NGRAPH_REPO=${NGRAPH_REPO}"
echo "    CMAKE_OPTIONS=${CMAKE_OPTIONS}"
echo "    GTEST_OUTPUT=${GTEST_OUTPUT}"

if [ -z ${CMD_TO_RUN} ] ; then
    echo "No CMD_TO_RUN specified - will run cmake, make, and style-check"

    echo "Running cmake"
    cmake ${CMAKE_OPTIONS} .. 2>&1 | tee ${OUTPUT_DIR}/cmake_${TEST_SUITE}.log
    echo "Running make"
    env VERBOSE=1 make -j ${PARALLEL} 2>&1 | tee ${OUTPUT_DIR}/make_${TEST_SUITE}.log

    # check style before running unit tests
    if [ -f "/usr/bin/clang-3.9" ]; then
        echo "Running make style-check"
        env VERBOSE=1 make -j style-check 2>&1 | tee ${OUTPUT_DIR}/make_style_check_${TEST_SUITE}.log
    fi

    if $RUN_UNIT_TESTS; then
        echo "Running make unit-test-check"
        env VERBOSE=1 make unit-test-check 2>&1 | tee ${OUTPUT_DIR}/make_unit_test_check_${TEST_SUITE}.log
    fi
else
    echo "Running make ${CMD_TO_RUN}"
    env VERBOSE=1 make ${CMD_TO_RUN} 2>&1 | tee ${OUTPUT_DIR}/make_${CMD_TO_RUN}_${TEST_SUITE}.log
fi

