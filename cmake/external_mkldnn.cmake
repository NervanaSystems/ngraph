# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

#----------------------------------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#----------------------------------------------------------------------------------------------------------

set(MKLDNN_BUILD_COMMAND_EXTRA_FLAGS ""
    CACHE STRING "Additional flags to supply to '${CMAKE_MAKE_PROGRAM}' when building MKLDNN."
    )

set(MKLDNN_CMAKE_EXTRA_FLAGS ""
    CACHE STRING "Additional flags to supply to 'cmake' when building the MKLDNN build."
    )

# CMake is a terrible language when it comes to lists and strings.  Here's the behavior we *want*:
#   1. CMake's user enters a space-separated list of additional command-line argments to be supplied
#      to the 'make' invocation that builds MKL-DNN.
#
#   2. When that 'make' invocation occurs, each of thoise command-line arguments is a separate token
#      on the command-line of the 'make' invocation.
#
# To avoid CMake grouping all of those command-line arguments together into a single, quote-
# delimited string on the 'make' command-line, we need to temporarily convert the user-specified,
# space-separated string into a CMake semicolon-separated list.
separate_arguments(MKLDNN_BUILD_COMMAND_EXTRA_FLAGS_LIST UNIX_COMMAND
    "${MKLDNN_BUILD_COMMAND_EXTRA_FLAGS}"
    )

separate_arguments(MKLDNN_CMAKE_EXTRA_FLAGS_LIST UNIX_COMMAND
    "${MKLDNN_CMAKE_EXTRA_FLAGS}"
    )

#----------------------------------------------------------------------------------------------------------

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    set(MKLDNN_GIT_REPO_URL https://github.com/01org/mkl-dnn)
    set(MKLDNN_GIT_TAG "144e0db")
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
                -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
                ${MKLDNN_CMAKE_EXTRA_FLAGS_LIST}
            BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" ${MKLDNN_BUILD_COMMAND_EXTRA_FLAGS_LIST}
            )
    else()
        ExternalProject_Add(
            ext_mkldnn
            GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
            GIT_TAG ${MKLDNN_GIT_TAG}
            UPDATE_COMMAND ""
            # Uncomment below with any in-flight MKL-DNN patches
            # PATCH_COMMAND patch -p1 < ${CMAKE_SOURCE_DIR}/third-party/patches/mkldnn-cmake-openmp.patch
            BUILD_BYPRODUCTS "${MKLDNN_INSTALL_DIR}/include/mkldnn.hpp"
            CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
                ${MKLDNN_CMAKE_EXTRA_FLAGS_LIST}
            BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" ${MKLDNN_BUILD_COMMAND_EXTRA_FLAGS_LIST}
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
