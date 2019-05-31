# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
# ONNX.proto definition version
#------------------------------------------------------------------------------

set(ONNX_VERSION 1.3.0)

#------------------------------------------------------------------------------
# Download and install libonnx ...
#------------------------------------------------------------------------------

set(ONNX_GIT_REPO_URL https://github.com/onnx/onnx.git)
set(ONNX_GIT_BRANCH rel-${ONNX_VERSION})

ExternalProject_Add(
    ext_onnx
    PREFIX onnx
    GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
    GIT_TAG ${ONNX_GIT_BRANCH}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DONNX_GEN_PB_TYPE_STUBS=OFF
               -DCMAKE_PREFIX_PATH=${Protobuf_INSTALL_PREFIX}
               -DONNX_ML=TRUE
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/bin"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx"
    EXCLUDE_FROM_ALL TRUE
)

# -----------------------------------------------------------------------------

ExternalProject_Get_Property(ext_onnx SOURCE_DIR BINARY_DIR)

set(ONNX_INCLUDE_DIR ${SOURCE_DIR}/onnx)
set(ONNX_PROTO_INCLUDE_DIR ${BINARY_DIR}/onnx)
if (WIN32)
    set(ONNX_LIBRARY ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx.lib)
    set(ONNX_PROTO_LIBRARY ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx_proto.lib)

    ExternalProject_Add_Step(
        ext_onnx
        CopyONNX
        COMMAND ${CMAKE_COMMAND} -E copy ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx.lib ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/onnx.lib
        COMMAND ${CMAKE_COMMAND} -E copy ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx_proto.lib ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/onnx_proto.lib
        COMMENT "Copy onnx libraries to ngraph build directory."
        DEPENDEES install
    )

else()
    set(ONNX_LIBRARY ${BINARY_DIR}/libonnx.a)
    set(ONNX_PROTO_LIBRARY ${BINARY_DIR}/libonnx_proto.a)
endif()
set(ONNX_LIBRARIES ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY})

if (NOT TARGET onnx::libonnx)
    add_library(onnx::libonnx UNKNOWN IMPORTED)
    set_target_properties(onnx::libonnx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_LIBRARY})
    add_dependencies(onnx::libonnx ext_onnx)
endif()

if (NOT TARGET onnx::libonnx_proto)
    add_library(onnx::libonnx_proto UNKNOWN IMPORTED)
    set_target_properties(onnx::libonnx_proto PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_PROTO_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_PROTO_LIBRARY})
    add_dependencies(onnx::libonnx_proto ext_onnx)
endif()
