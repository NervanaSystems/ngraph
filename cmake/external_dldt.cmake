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


include(ExternalProject)

set(DLDT_REPO_URL https://github.com/opencv/dldt.git)


# Change these commit IDs to move to latest stable versions
set(DLDT_COMMIT_ID b2140c083a068a63591e8c2e9b5f6b240790519d)

# Add ngraph building to cmake flags of dldt
# This inner ngraph is expected to build with NGRAPH_OPV_ENABLE off
list(APPEND DLDT_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DENABLE_INFERENCE_ENGINE=ON
        -DENABLE_NGRAPH=ON
        -DNGRAPH_ONNX_IMPORT_ENABLE=ON
        -DNGRAPH_JSON_ENABLE=ON
        -DENABLE_CLDNN=OFF
    )


ExternalProject_Add(
    ext_dldt
    PREFIX dldt
    GIT_REPOSITORY ${DLDT_REPO_URL}
    GIT_TAG ${DLDT_COMMIT_ID}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}
               -DCMAKE_CXX_FLAGS=${CMAKE_ORIGINAL_CXX_FLAGS}
               -DCMAKE_PREFIX_PATH=${Protobuf_INSTALL_PREFIX}
               ${DLDT_CMAKE_ARGS}
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt/bin"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/dldt"
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_dldt SOURCE_DIR)
add_library(libdldt INTERFACE)
target_include_directories(libdldt SYSTEM INTERFACE ${SOURCE_DIR}/inference-engine/include)
add_dependencies(libdldt ext_dldt)
