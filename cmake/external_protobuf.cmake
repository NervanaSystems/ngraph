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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

set(PROTOBUF_GIT_REPO_URL https://github.com/google/protobuf.git)
set(PROTOBUF_GIT_BRANCH origin/3.5.x)

# The 'BUILD_BYPRODUCTS' arguments was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${PROTOBUF_GIT_REPO_URL}
        GIT_TAG ${PROTOBUF_GIT_BRANCH}
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ./autogen.sh && ./configure --disable-shared CXXFLAGS=-fPIC
        BUILD_COMMAND ${MAKE}
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
        EXCLUDE_FROM_ALL TRUE
        )
else()
    if (${CMAKE_VERSION} VERSION_LESS 3.6)
        ExternalProject_Add(
            ext_protobuf
            PREFIX ext_protobuf
            GIT_REPOSITORY ${PROTOBUF_GIT_REPO_URL}
            GIT_TAG ${PROTOBUF_GIT_BRANCH}
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND ./autogen.sh && ./configure --disable-shared CXXFLAGS=-fPIC
            BUILD_COMMAND ${MAKE}
            TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
            STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
            DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
            SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
            BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/protobuf/src/src/.libs/libprotobuf.a"
            EXCLUDE_FROM_ALL TRUE
            )
    else()
        # To speed things up prefer 'shallow copy' for CMake 3.6 and later
        ExternalProject_Add(
            ext_protobuf
            PREFIX ext_protobuf
            GIT_REPOSITORY ${PROTOBUF_GIT_REPO_URL}
            GIT_TAG ${PROTOBUF_GIT_BRANCH}
            GIT_SHALLOW TRUE
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND ./autogen.sh && ./configure --disable-shared CXXFLAGS=-fPIC
            BUILD_COMMAND ${MAKE}
            TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
            STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
            DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
            SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
            BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/protobuf/src/src/.libs/libprotobuf.a"
            EXCLUDE_FROM_ALL TRUE
            )
    endif()
endif()

# -----------------------------------------------------------------------------

ExternalProject_Get_Property(ext_protobuf SOURCE_DIR BINARY_DIR)

# -----------------------------------------------------------------------------
# Use the interface of FindProtobuf.cmake
# -----------------------------------------------------------------------------

set(Protobuf_SRC_ROOT_FOLDER ${SOURCE_DIR})

set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_SRC_ROOT_FOLDER}/src/protoc)
set(Protobuf_INCLUDE_DIR ${Protobuf_SRC_ROOT_FOLDER}/src)
set(Protobuf_LIBRARY ${Protobuf_SRC_ROOT_FOLDER}/src/.libs/libprotobuf.a)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARY})

if (NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
    set_target_properties(protobuf::libprotobuf PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}"
        IMPORTED_LOCATION "${Protobuf_LIBRARY}")
    add_dependencies(protobuf::libprotobuf ext_protobuf)
endif()

if (NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
    add_dependencies(protobuf::protoc ext_protobuf)
endif()

set(Protobuf_FOUND)
set(PROTOBUF_FOUND)
