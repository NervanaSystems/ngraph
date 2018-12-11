# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

if (NOT NGRAPH_MANYLINUX_ENABLE)
    return()
endif()

#
# OpenMP runtime bundled with mklml cannot run with GLIBC==2.5
# For manylinux1, build the OpenMP runtime from LLVM and use it instead.
#

include(ExternalProject)

set(OMPRT_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/omprt)

ExternalProject_Add(
    ext_omprt
    DEPENDS ext_mkldnn
    GIT_REPOSITORY https://github.com/llvm-mirror/openmp.git
    GIT_TAG 366ce74b85790ed41f94fba7f17a0911bde83500
    CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${OMPRT_INSTALL_PREFIX}
            -DLIBOMP_OMPT_SUPPORT=OFF
    TMP_DIR "${OMPRT_INSTALL_PREFIX}/tmp"
    STAMP_DIR "${OMPRT_INSTALL_PREFIX}/stamp"
    DOWNLOAD_DIR "${OMPRT_INSTALL_PREFIX}/download"
    SOURCE_DIR "${OMPRT_INSTALL_PREFIX}/src"
    BINARY_DIR "${OMPRT_INSTALL_PREFIX}/build"
    INSTALL_DIR "${OMPRT_INSTALL_PREFIX}"
    EXCLUDE_FROM_ALL TRUE
)

add_custom_command(TARGET ext_omprt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${OMPRT_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}omp${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${OMPRT_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}gomp${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${OMPRT_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_BUILD_DIR}
    COMMENT "Move OpenMP runtime libraries to ngraph build directory."
)
