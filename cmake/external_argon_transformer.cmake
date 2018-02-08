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

# To build ngraph with Argon transformer using pre-build Argon API
# ```
# cmake -DNGRAPH_ARGON_ENABLE=True -DNGRAPH_PREBUILD_ARGON_API_PATH=$HOME/dev/system/_out/debug-x86_64-Linux ..
# make -j
# ```

# Enable ExternalProject CMake module
include(ExternalProject)

if (NGRAPH_ARGON_ENABLE)
    # We require pre-build Argon API library
    if (NOT DEFINED NGRAPH_PREBUILD_ARGON_API_PATH)
        message(FATAL_ERROR "NGRAPH_PREBUILD_ARGON_API_PATH not defined, set it with -DNGRAPH_PREBUILD_ARGON_API_PATH=")
    endif()

    # Repository
    set(ARGON_TRANSFORMER_CMAKE_GIT_REPOSITORY git@github.com:NervanaSystems/argon-transformer.git)

    # Set argon_transformer tag
    # Notes:
    # - Before we have ngraph CI job for argon transformer, ngraph master might not be
    #   compatible with argon transformer. To ensure compatibility, checkout the ngraph commit point
    #   where the following `ARGON_TRANSFORMER_CMAKE_GIT_TAG` is set and build ngraph with argon using this
    #   commit.
    # - After we have ngraph CI job for argon transformer, ngraph master will be compatible with
    #   argon transformer guaranteed by CI.
    set(ARGON_TRANSFORMER_CMAKE_GIT_TAG cpp-master)

    # Determines where argon-transformer will be located
    set(ARGON_TRANSFORMER_CMAKE_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/argon_transformer)

    # Print
    message(STATUS "NGRAPH_INCLUDE_PATH: ${NGRAPH_INCLUDE_PATH}")
    message(STATUS "LLVM_INCLUDE_DIR: ${LLVM_INCLUDE_DIR}")
    message(STATUS "NGRAPH_PREBUILD_ARGON_API_PATH: ${NGRAPH_PREBUILD_ARGON_API_PATH}")

    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2
    if (${CMAKE_VERSION} VERSION_LESS 3.2)
        if (DEFINED CUSTOM_ARGON_TRANSFORMER_DIR)
            ExternalProject_Add(
                ext_argon_transformer
                SOURCE_DIR ${CUSTOM_ARGON_TRANSFORMER_DIR}
                PREFIX ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DNGRAPH_INSTALL_PREFIX=${ARGON_TRANSFORMER_CMAKE_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                BUILD_ALWAYS 1
            )
        else()
            ExternalProject_Add(
                ext_argon_transformer
                GIT_REPOSITORY ${ARGON_TRANSFORMER_CMAKE_GIT_REPOSITORY}
                GIT_TAG ${ARGON_TRANSFORMER_CMAKE_GIT_TAG}
                PREFIX ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DNGRAPH_INSTALL_PREFIX=${ARGON_TRANSFORMER_CMAKE_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                BUILD_ALWAYS 1
            )
        endif()
    else()
        if (DEFINED CUSTOM_ARGON_TRANSFORMER_DIR)
            ExternalProject_Add(
                ext_argon_transformer
                SOURCE_DIR ${CUSTOM_ARGON_TRANSFORMER_DIR}
                PREFIX ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DNGRAPH_INSTALL_PREFIX=${ARGON_TRANSFORMER_CMAKE_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                BUILD_BYPRODUCTS ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                BUILD_ALWAYS 1
            )
        else()
            ExternalProject_Add(
                ext_argon_transformer
                GIT_REPOSITORY ${ARGON_TRANSFORMER_CMAKE_GIT_REPOSITORY}
                GIT_TAG ${ARGON_TRANSFORMER_CMAKE_GIT_TAG}
                PREFIX ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DNGRAPH_INSTALL_PREFIX=${ARGON_TRANSFORMER_CMAKE_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                BUILD_BYPRODUCTS ${ARGON_TRANSFORMER_CMAKE_PREFIX}
                BUILD_ALWAYS 1
            )
        endif()
    endif()

    ExternalProject_Get_Property(ext_argon_transformer source_dir)
    set(ARGON_TRANSFORMER_INCLUDE_DIR ${ARGON_TRANSFORMER_CMAKE_PREFIX}/include PARENT_SCOPE)
    set(ARGON_TRANSFORMER_LIB_DIR ${ARGON_TRANSFORMER_CMAKE_PREFIX}/lib PARENT_SCOPE)
    set(ARGON_API_INCLUDE_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/include PARENT_SCOPE)
    set(ARGON_API_LIB_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/lib) # Used by find_library below
    set(ARGON_API_LIB_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/lib PARENT_SCOPE)

    # Find prebuild argon library
    find_library(ARGON_API_LIBS
        NAMES
        argon_api
        optimizer
        ir_builder
        umd
        flex
        disasm
        HINTS
        ${ARGON_API_LIB_DIR}
        $ENV{LD_LIBRARY_PATH}
    )

endif()
