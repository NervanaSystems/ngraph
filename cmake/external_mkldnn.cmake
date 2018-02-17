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

include(ExternalProject)

#----------------------------------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#----------------------------------------------------------------------------------------------------------

if(NGRAPH_CPU_ENABLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    set(MKLDNN_GIT_REPO_URL https://github.com/intel/mkl-dnn)
    set(MKLDNN_GIT_TAG "3e1f8f5")
    set(MKLDNN_INSTALL_DIR ${EXTERNAL_INSTALL_DIR}/mkldnn)

    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
    if(${CMAKE_VERSION} VERSION_LESS 3.2)
        ExternalProject_Add(
            ext_mkldnn
            GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
            GIT_TAG ${MKLDNN_GIT_TAG}
            UPDATE_COMMAND ""
            # Uncomment below with any in-flight MKL-DNN patches
            # PATCH_COMMAND patch -p1 < ${CMAKE_SOURCE_DIR}/third-party/patches/mkldnn-cmake-openmp.patch
            CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
            )
    else()
        ExternalProject_Add(
            ext_mkldnn
            GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
            GIT_TAG ${MKLDNN_GIT_TAG}
            UPDATE_COMMAND ""
            # Uncomment below with any in-flight MKL-DNN patches
            # PATCH_COMMAND patch -p1 < ${CMAKE_SOURCE_DIR}/third-party/patches/mkldnn-cmake-openmp.patch
            CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
            BUILD_BYPRODUCTS "${MKLDNN_INSTALL_DIR}/include/mkldnn.hpp"
            )
    endif()

    ExternalProject_Get_Property(ext_mkldnn source_dir binary_dir)

    ExternalProject_Add_Step(
        ext_mkldnn
        PrepareMKL
        COMMAND ${source_dir}/scripts/prepare_mkl.sh
        DEPENDEES download
        DEPENDERS configure
        )


    set(MKLDNN_INCLUDE_DIR "${MKLDNN_INSTALL_DIR}/include" PARENT_SCOPE)
    set(MKLDNN_LIB_DIR "${MKLDNN_INSTALL_DIR}/lib" PARENT_SCOPE)

endif()
