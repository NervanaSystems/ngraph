# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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

if(NOT NGRAPH_MANYLINUX_ENABLE)
    return()
endif()

#
# OpenMP runtime bundled with mklml cannot run with GLIBC==2.5
# For manylinux1, build the OpenMP runtime from LLVM and use it instead.
#

include(cmake/external_hwloc.cmake)

include(ExternalProject)

set(NGRAPH_LLVM_OMPRT_VERSION 8.0.0)

set(OMPRT_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/omprt)

ExternalProject_Add(
    ext_omprt
    PREFIX omprt
    DEPENDS ext_hwloc
    URL http://releases.llvm.org/${NGRAPH_LLVM_OMPRT_VERSION}/openmp-${NGRAPH_LLVM_OMPRT_VERSION}.src.tar.xz
    URL_HASH SHA1=90462a0f720a9a40ecbda9636c24d627b5dc05db
    DOWNLOAD_NO_PROGRESS TRUE
    PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/omprt.patch
    CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${OMPRT_INSTALL_PREFIX}
            -DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
            -DLIBOMP_OMPT_SUPPORT=OFF
            -DLIBOMP_LIB_NAME=${CMAKE_SHARED_LIBRARY_PREFIX}iomp5
            -DLIBOMP_INSTALL_ALIASES=OFF
            -DLIBOMP_USE_HWLOC=ON
            -DLIBOMP_HWLOC_INSTALL_DIR=${HWLOC_INSTALL_PREFIX}
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_omprt INSTALL_DIR)

ExternalProject_Add_Step(
    ext_omprt
    CopyOMPRT
    COMMAND ${CMAKE_COMMAND} -E copy ${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX}
    COMMENT "Copy OpenMP runtime libraries to ngraph build directory."
    DEPENDEES install
    )
