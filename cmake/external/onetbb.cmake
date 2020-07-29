# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

#------------------------------------------------------------------------------
# Fetch and configure TBB
#------------------------------------------------------------------------------

if(NOT NGRAPH_TBB_ENABLE)
    return()
endif()

cmake_policy(SET CMP0074 NEW)

set(NGRAPH_TBB_VERSION "2020.2")

if (WIN32)
    set(TBB_FILE https://github.com/oneapi-src/oneTBB/releases/download/v${NGRAPH_TBB_VERSION}/tbb-${NGRAPH_TBB_VERSION}-win.zip)
    set(TBB_SHA1_HASH 38b2af1626e5dea06269c17ffb85f190d4f9b79a)
elseif(APPLE)
    set(TBB_FILE https://github.com/oneapi-src/oneTBB/releases/download/v${NGRAPH_TBB_VERSION}/tbb-${NGRAPH_TBB_VERSION}-mac.tgz)
    set(TBB_SHA1_HASH 19b56f90bae806e7c9a9f331bb03db934f046016)
endif()

include(FetchContent)

message(STATUS "Fetching oneTBB")

if(WIN32 OR APPLE)
    FetchContent_Declare(
        ngraphtbb
        URL            ${TBB_FILE}
        URL_HASH       SHA1=${TBB_SHA1_HASH}
    )
else()
    set(TBB_GIT_REPO_URL https://github.com/oneapi-src/oneTBB.git)
    FetchContent_Declare(
        ngraphtbb
        GIT_REPOSITORY ${TBB_GIT_REPO_URL}
        GIT_TAG        v${NGRAPH_TBB_VERSION}
        GIT_SHALLOW    1
    )
endif()

FetchContent_GetProperties(ngraphtbb)
if(NOT ngraphtbb_POPULATED)
    FetchContent_Populate(ngraphtbb)
endif()

if(WIN32 OR APPLE)
    set(TBB_DIR  ${ngraphtbb_SOURCE_DIR}/tbb/cmake)
else()
    set(TBB_ROOT ${ngraphtbb_SOURCE_DIR})
    include(${TBB_ROOT}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS tbb_build_dir=${PROJECT_BINARY_DIR}/tbb_build
        tbb_build_prefix=tbb stdver=c++${NGRAPH_CXX_STANDARD} CONFIG_DIR TBB_DIR)
endif()

find_package(TBB REQUIRED tbb)
if (NOT TBB_FOUND)
    message(FATAL_ERROR "TBB is needed by the CPU backend and was not found")
else()
    message(STATUS "Found TBB and imported target ${TBB_IMPORTED_TARGETS}")
endif()

target_compile_definitions(TBB::tbb INTERFACE TBB_USE_THREADING_TOOLS)

set_source_files_properties(cpu_external_function.cpp
    PROPERTIES COMPILE_DEFINITIONS "NGRAPH_TBB_ENABLE")

install(FILES $<TARGET_FILE:TBB::tbb>
    DESTINATION ${NGRAPH_INSTALL_LIB})
