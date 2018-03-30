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

cmake_minimum_required (VERSION 3.1)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

project (pyngraph)

include(ExternalProject)

ExternalProject_Add(
    pybind11
    GIT_REPOSITORY "https://github.com/jagerman/pybind11.git"
    GIT_TAG "allow-nonconstructible-holders"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/pybind11"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

#
# The user can always override this by using the cmake command listed below.
if (DEFINED NGRAPH_INSTALL_PREFIX)
    if(NOT EXISTS ${NGRAPH_INSTALL_PREFIX}/include/ngraph)
        message(FATAL_ERROR "Cannot find ngraph library in ${NGRAPH_INSTALL_PREFIX} make sure that NGRAPH_INSTALL_PREFIX is set correctly")
    endif()
    set(USE_EMBEDDED_NGRAPH FALSE)
else()
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/../src/ngraph)
        message(FATAL_ERROR "Cannot find ngraph source in ${CMAKE_SOURCE_DIR}/..")
    endif()
    set(USE_EMBEDDED_NGRAPH TRUE)
    set(NGRAPH_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/ngraph_dist)
endif()

if(USE_EMBEDDED_NGRAPH)
    set(NGRAPH_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)
    set(NGRAPH_CMAKE_ARGS "-DNGRAPH_INSTALL_PREFIX=${NGRAPH_INSTALL_PREFIX}")

    if(APPLE)
        set(NGRAPH_CMAKE_ARGS "-DCMAKE_MACOSX_RPATH=ON" ${NGRAPH_CMAKE_ARGS})
    endif()

    ExternalProject_Add(
        ngraph
        DOWNLOAD_COMMAND ""
        SOURCE_DIR ${NGRAPH_SOURCE_DIR}
        CMAKE_ARGS ${NGRAPH_CMAKE_ARGS}
        )
endif()
