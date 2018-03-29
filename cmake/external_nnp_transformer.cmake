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

# To build ngraph with NNP transformer using pre-build Argon API
# ```
# cmake -DNGRAPH_NNP_ENABLE=True -DNGRAPH_PREBUILD_ARGON_API_PATH=$HOME/dev/system/_out/debug-x86_64-Linux ..
# make -j
# ```

# Enable ExternalProject CMake module
include(ExternalProject)

if (NGRAPH_NNP_ENABLE)
    # We require pre-build Argon API library
    if (NOT DEFINED NGRAPH_PREBUILD_ARGON_API_PATH)
        message(FATAL_ERROR "NGRAPH_PREBUILD_ARGON_API_PATH not defined, set it with -DNGRAPH_PREBUILD_ARGON_API_PATH=")
    endif()

    # Repository
    if (DEFINED CUSTOM_NNP_TRANSFORMER_GIT_REPOSITORY)
        set(NNP_TRANSFORMER_GIT_REPOSITORY ${CUSTOM_NNP_TRANSFORMER_GIT_REPOSITORY})
    else()
        set(NNP_TRANSFORMER_GIT_REPOSITORY https://github.com/NervanaSystems/nnp-transformer.git)
    endif()

    # Set nnp_transformer tag
    # Notes:
    # - Before we have ngraph CI job for nnp transformer, ngraph master might not be
    #   compatible with nnp transformer. To ensure compatibility, checkout the ngraph commit point
    #   where the following `NNP_TRANSFORMER_GIT_TAG` is set and build ngraph with nnp using this
    #   commit.
    # - After we have ngraph CI job for nnp transformer, ngraph master will be compatible with
    #   nnp transformer guaranteed by CI.
    set(NNP_TRANSFORMER_GIT_TAG master)

    # Determines where nnp-transformer will be located
    set(NNP_TRANSFORMER_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/nnp_transformer)

    # Print
    message(STATUS "NGRAPH_INCLUDE_PATH: ${NGRAPH_INCLUDE_PATH}")
    message(STATUS "LLVM_INCLUDE_DIR: ${LLVM_INCLUDE_DIR}")
    message(STATUS "NGRAPH_PREBUILD_ARGON_API_PATH: ${NGRAPH_PREBUILD_ARGON_API_PATH}")

    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2
    if (${CMAKE_VERSION} VERSION_LESS 3.2)
        if (DEFINED CUSTOM_NNP_TRANSFORMER_DIR)
            ExternalProject_Add(
                ext_nnp_transformer
                SOURCE_DIR ${CUSTOM_NNP_TRANSFORMER_DIR}
                PREFIX ${NNP_TRANSFORMER_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DNGRAPH_INSTALL_PREFIX=${NNP_TRANSFORMER_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                        -DMKLDNN_INCLUDE_DIR=${MKLDNN_INCLUDE_DIR}
                BUILD_ALWAYS 1
            )
        else()
            ExternalProject_Add(
                ext_nnp_transformer
                GIT_REPOSITORY ${NNP_TRANSFORMER_GIT_REPOSITORY}
                GIT_TAG ${NNP_TRANSFORMER_GIT_TAG}
                PREFIX ${NNP_TRANSFORMER_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DNGRAPH_INSTALL_PREFIX=${NNP_TRANSFORMER_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                        -DMKLDNN_INCLUDE_DIR=${MKLDNN_INCLUDE_DIR}
                BUILD_ALWAYS 1
            )
        endif()
    else()
        if (DEFINED CUSTOM_NNP_TRANSFORMER_DIR)
            ExternalProject_Add(
                ext_nnp_transformer
                SOURCE_DIR ${CUSTOM_NNP_TRANSFORMER_DIR}
                PREFIX ${NNP_TRANSFORMER_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DNGRAPH_INSTALL_PREFIX=${NNP_TRANSFORMER_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                        -DMKLDNN_INCLUDE_DIR=${MKLDNN_INCLUDE_DIR}
                BUILD_BYPRODUCTS ${NNP_TRANSFORMER_PREFIX}
                BUILD_ALWAYS 1
            )
        else()
            ExternalProject_Add(
                ext_nnp_transformer
                GIT_REPOSITORY ${NNP_TRANSFORMER_GIT_REPOSITORY}
                GIT_TAG ${NNP_TRANSFORMER_GIT_TAG}
                PREFIX ${NNP_TRANSFORMER_PREFIX}
                UPDATE_COMMAND ""
                CMAKE_ARGS
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DNGRAPH_INSTALL_PREFIX=${NNP_TRANSFORMER_PREFIX}
                        -DPREBUILD_ARGON_API_PATH=${NGRAPH_PREBUILD_ARGON_API_PATH}
                        -DEXTERNAL_NGRAPH_INCLUDE_DIR=${NGRAPH_INCLUDE_PATH}
                        -DINSTALLED_HEADERS_PATH=${CMAKE_INSTALL_PREFIX}/include
                        -DMKLDNN_INCLUDE_DIR=${MKLDNN_INCLUDE_DIR}
                BUILD_BYPRODUCTS ${NNP_TRANSFORMER_PREFIX}
                BUILD_ALWAYS 1
            )
        endif()
    endif()

    ExternalProject_Get_Property(ext_nnp_transformer source_dir)
    set(NNP_TRANSFORMER_SOURCE_DIR ${source_dir} PARENT_SCOPE)
    set(NNP_TRANSFORMER_INCLUDE_DIR ${NNP_TRANSFORMER_PREFIX}/include PARENT_SCOPE)
    set(NNP_TRANSFORMER_LIB_DIR ${NNP_TRANSFORMER_PREFIX}/lib PARENT_SCOPE)
    set(ARGON_API_INCLUDE_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/include PARENT_SCOPE)
    set(ARGON_API_LIB_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/lib) # Used by find_library below
    set(ARGON_API_LIB_DIR ${NGRAPH_PREBUILD_ARGON_API_PATH}/lib PARENT_SCOPE)

    # Find prebuild nnp library
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
