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

# Enable FetchContent CMake module
include(FetchContent)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

message(STATUS "Fetching Google Protobuf")

# This version of PROTOBUF is required by Microsoft ONNX Runtime.
set(NGRAPH_PROTOBUF_GIT_REPO_URL "https://github.com/protocolbuffers/protobuf")

if(CMAKE_CROSSCOMPILING)
    # Cross Compiling
    # Protobuf source version has to match system protoc version
    # Find system protoc and version
    # Setup extra protobuf build flags for cross compiling
    find_program(SYSTEM_PROTOC protoc PATHS ENV PATH)

    if(SYSTEM_PROTOC)
        execute_process(COMMAND ${SYSTEM_PROTOC} --version OUTPUT_VARIABLE PROTOC_VERSION)
        string(REPLACE " " ";" PROTOC_VERSION ${PROTOC_VERSION})
        list(GET PROTOC_VERSION -1 PROTOC_VERSION)
        message("Detected system protoc version: ${PROTOC_VERSION}")

        if(${PROTOC_VERSION} VERSION_EQUAL "3.0.0")
            message(WARNING "Protobuf 3.0.0 detected switching to 3.0.2 due to bug in gmock url")
            set(PROTOC_VERSION "3.0.2")
        endif()

        set(PROTOBUF_SYSTEM_PROTOC --with-protoc=${SYSTEM_PROTOC})
        set(PROTOBUF_SYSTEM_PROCESSOR --host=${CMAKE_HOST_SYSTEM_PROCESSOR})
    else()
        message(FATAL_ERROR "System Protobuf is needed while cross-compiling")
    endif()
else()
    set(PROTOC_VERSION "3.11.3")
endif()

set(NGRAPH_PROTOBUF_GIT_TAG "v${PROTOC_VERSION}")

FetchContent_Declare(
    ext_protobuf
    GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
    GIT_TAG        ${NGRAPH_PROTOBUF_GIT_TAG}
    GIT_SHALLOW    1
)

FetchContent_GetProperties(ext_protobuf)
if(NOT ext_protobuf_POPULATED)
    FetchContent_Populate(ext_protobuf)
endif()

include(ProcessorCount)
ProcessorCount(N)
if(N EQUAL 0)
    set(N 8)
endif()

# Two ways for building protobuf
# 1. CMake ( WIN32 or cross compiling )
# 2. autogen.sh -> configure -> make
if(WIN32 OR (NOT WIN32 AND NOT APPLE AND (PROTOC_VERSION VERSION_GREATER "3.0") AND CMAKE_CROSSCOMPILING))
    set(PROTOBUF_CMAKE_ARGS
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        -DCMAKE_CXX_FLAGS=${CMAKE_ORIGINAL_CXX_FLAGS}
        -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
        -Dprotobuf_WITH_ZLIB=OFF
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/protobuf)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
        -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
        ${PROTOBUF_CMAKE_ARGS}
        "${ext_protobuf_SOURCE_DIR}/cmake"
        WORKING_DIRECTORY "${ext_protobuf_BINARY_DIR}")
    if("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
        execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${N}
            WORKING_DIRECTORY "${ext_protobuf_BINARY_DIR}")
    else()
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${ext_protobuf_BINARY_DIR}")
    endif()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${EXTERNAL_PROJECTS_ROOT}/protobuf)
else()
    if ((NOT APPLE) AND DEFINED NGRAPH_USE_CXX_ABI)
        set(BUILD_FLAGS "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    else()
        set(BUILD_FLAGS "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC")
    endif()
    execute_process(COMMAND ./autogen.sh WORKING_DIRECTORY ${ext_protobuf_SOURCE_DIR})
    # Don't manually set compiler on macos since it causes compile error on macos >= 10.14
    if (APPLE)
        execute_process(COMMAND ${ext_protobuf_SOURCE_DIR}/configure ${PROTOBUF_SYSTEM_PROTOC} ${PROTOBUF_SYSTEM_PROCESSOR} SDKROOT=${CMAKE_OSX_SYSROOT} --prefix=${ext_protobuf_BINARY_DIR} --disable-shared WORKING_DIRECTORY ${ext_protobuf_BINARY_DIR})
        execute_process(COMMAND make -j${N} SDKROOT=${CMAKE_OSX_SYSROOT} "${BUILD_FLAGS}" install WORKING_DIRECTORY ${ext_protobuf_BINARY_DIR})
    else()
        execute_process(COMMAND ${ext_protobuf_SOURCE_DIR}/configure ${PROTOBUF_SYSTEM_PROTOC} ${PROTOBUF_SYSTEM_PROCESSOR} CXX=${CMAKE_CXX_COMPILER} --prefix=${ext_protobuf_BINARY_DIR} --disable-shared WORKING_DIRECTORY ${ext_protobuf_BINARY_DIR})
        execute_process(COMMAND make -j${N} "${BUILD_FLAGS}" install WORKING_DIRECTORY ${ext_protobuf_BINARY_DIR})
    endif()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ext_protobuf_BINARY_DIR})
endif()
find_package(Protobuf ${PROTOC_VERSION} REQUIRED)
if (NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf is needed but was not found")
endif()
