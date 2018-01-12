# Copyright 2017 Nervana Systems Inc.
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

# To build ngraph with Argon backend
# * Option 1: Build argon-api from scratch
#   ```
#   cmake -DNGRAPH_ARGON_ENABLE=True ..
#   make -j
#   ```
# * Option 2: Using pre-build argon-api
#   Use PREBUILD_ARGON_PATH to specify the pre-build argon-api path
#   ```
#   cmake -DNGRAPH_ARGON_ENABLE=True -DPREBUILD_ARGON_PATH=$HOME/dev/system/_out/debug-x86_64-Linux ..
#   make -j
#   ```

# Enable ExternalProject CMake module
include(ExternalProject)

if (NGRAPH_ARGON_ENABLE)
    # Repository
    set(ARGON_CMAKE_GIT_REPOSITORY git@github.com:NervanaSystems/argon-transformer.git)

    # Set argon_transformer tag
    # Notes:
    # - Before we have ngraph CI job for argon transformer, ngraph master might not be
    #   compatible with argon transformer. To ensure compatibility, checkout the ngraph commit point
    #   where the following `ARGON_CMAKE_GIT_TAG` is set and build ngraph with argon using this
    #   commit.
    # - After we have ngraph CI job for argon transformer, ngraph master will be compatible with
    #   argon transformer guaranteed by CI.
    set(ARGON_CMAKE_GIT_TAG a0fe149440d1a2b10fb33ecd2e05563c7e42d8b9) # Thu Jan 11 2018

    set(ARGON_CMAKE_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/argon)
    if (NOT DEFINED PREBUILD_ARGON_PATH)
        set(PREBUILD_ARGON_PATH "")
    endif()

    # Print
    message(STATUS "NGRAPH_INCLUDE_PATH: ${NGRAPH_INCLUDE_PATH}")
    message(STATUS "LLVM_INCLUDE_DIR: ${LLVM_INCLUDE_DIR}")
    message(STATUS "PREBUILD_ARGON_PATH: ${PREBUILD_ARGON_PATH}")

    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2
    if (${CMAKE_VERSION} VERSION_LESS 3.2)
        ExternalProject_Add(
            ext_argon
            GIT_REPOSITORY ${ARGON_CMAKE_GIT_REPOSITORY}
            GIT_TAG ${ARGON_CMAKE_GIT_TAG}
            PREFIX ${ARGON_CMAKE_PREFIX}
            UPDATE_COMMAND ""
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                       -DCMAKE_INSTALL_PREFIX=${ARGON_CMAKE_PREFIX}/src/ext_argon-build/argon
                       -DPREBUILD_ARGON_PATH=${PREBUILD_ARGON_PATH}
                       -DARGON_AS_EXTERNAL=True
                       -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                       -DLLVM_INCLUDE_DIR=${LLVM_INCLUDE_DIR}
            BUILD_ALWAYS 1
        )
    else()
        ExternalProject_Add(
            ext_argon
            GIT_REPOSITORY ${ARGON_CMAKE_GIT_REPOSITORY}
            GIT_TAG ${ARGON_CMAKE_GIT_TAG}
            PREFIX ${ARGON_CMAKE_PREFIX}
            UPDATE_COMMAND ""
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                       -DPREBUILD_ARGON_PATH=${PREBUILD_ARGON_PATH}
                       -DCMAKE_INSTALL_PREFIX=${ARGON_CMAKE_PREFIX}/src/ext_argon-build/argon
                       -DARGON_AS_EXTERNAL=True
                       -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                       -DLLVM_INCLUDE_DIR=${LLVM_INCLUDE_DIR}
            BUILD_BYPRODUCTS ${ARGON_CMAKE_PREFIX}
            BUILD_ALWAYS 1
        )
    endif()

    ExternalProject_Get_Property(ext_argon source_dir binary_dir)
    set(ARGON_INCLUDE_DIR "${source_dir}/argon/src" PARENT_SCOPE)
    set(ARGON_LIB_DIR "${binary_dir}/argon" PARENT_SCOPE)
endif()
