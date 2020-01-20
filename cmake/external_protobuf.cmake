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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

# This version of PROTOBUF is required by Microsoft ONNX Runtime.
set(NGRAPH_PROTOBUF_GIT_REPO_URL "https://github.com/protocolbuffers/protobuf")

if(NGRAPH_ONNX_IMPORT_ENABLE)
    set(NGRAPH_PROTOBUF_GIT_TAG "v3.5.2")
else()
    set(NGRAPH_PROTOBUF_GIT_TAG "v3.6.1")
endif()

set(Protobuf_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/protobuf)
set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_INSTALL_PREFIX}/bin/protoc)
set(Protobuf_INCLUDE_DIR ${Protobuf_INSTALL_PREFIX}/include)
if (WIN32)
    set(Protobuf_LIBRARY ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.lib)
else()
    set(Protobuf_LIBRARY ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.a)
endif()

if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(MAKE_UTIL make)
else()
    set(MAKE_UTIL $(MAKE))
endif()

if (WIN32)
    ExternalProject_Add(
        ext_protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
        GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_CXX_FLAGS=${CMAKE_ORIGINAL_CXX_FLAGS}
            -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
            -Dprotobuf_WITH_ZLIB=OFF
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/protobuf
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        SOURCE_SUBDIR "cmake"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS ${Protobuf_PROTOC_EXECUTABLE} ${Protobuf_LIBRARY}
    )
elseif (APPLE)
    # Don't manually set compiler on macos since it causes compile error on macos >= 10.14
    ExternalProject_Add(
        ext_protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
        GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${EXTERNAL_PROJECTS_ROOT}/protobuf --disable-shared
        BUILD_COMMAND $(MAKE) "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC"
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS ${Protobuf_PROTOC_EXECUTABLE} ${Protobuf_LIBRARY}
        )
else()
    if (DEFINED NGRAPH_USE_CXX_ABI)
        set(BUILD_FLAGS "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    else()
        set(BUILD_FLAGS "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC")
    endif()

    ExternalProject_Add(
        ext_protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
        GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${EXTERNAL_PROJECTS_ROOT}/protobuf --disable-shared CXX=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ${MAKE_UTIL} "${BUILD_FLAGS}"
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS ${Protobuf_PROTOC_EXECUTABLE} ${Protobuf_LIBRARY}
        )
endif()

# -----------------------------------------------------------------------------
# Use the interface of FindProtobuf.cmake
# -----------------------------------------------------------------------------

if(NGRAPH_ONNX_IMPORT_ENABLE)
    if (NOT TARGET libprotobuf)
        add_library(libprotobuf INTERFACE)
        if (WIN32)
            target_link_libraries(libprotobuf INTERFACE
                debug ${Protobuf_INSTALL_PREFIX}/lib/libprotobufd.lib
                optimized ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.lib)
        else()
            target_link_libraries(libprotobuf INTERFACE
                ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.a)
        endif()
        set_target_properties(libprotobuf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
        add_dependencies(libprotobuf ext_protobuf)
    endif()
    set(Protobuf_LIBRARIES libprotobuf)
else()
    if (NOT TARGET protobuf::libprotobuf)
        add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
        set_target_properties(protobuf::libprotobuf PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}"
            IMPORTED_LOCATION "${Protobuf_LIBRARY}")
        add_dependencies(protobuf::libprotobuf ext_protobuf)
    endif()
    set(Protobuf_LIBRARIES protobuf::libprotobuf)
endif()

if (NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED)
    set_target_properties(protobuf::protoc PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Protobuf_PROTOC_EXECUTABLE}"
        IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
    add_dependencies(protobuf::protoc ext_protobuf)
endif()

set(Protobuf_FOUND TRUE)
set(PROTOBUF_FOUND TRUE)
