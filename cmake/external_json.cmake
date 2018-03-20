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
# Download json
#------------------------------------------------------------------------------

SET(JSON_GIT_REPO_URL https://github.com/nlohmann/json)
SET(JSON_GIT_LABEL v3.1.1)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_json
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        # Disable install step
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        # cmake does not allow calling cmake functions so we call a cmake script in the Module
        # directory.
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/Modules/patch_json.cmake
        )
else()
    ExternalProject_Add(
        ext_json
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        # Disable install step
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        # cmake does not allow calling cmake functions so we call a cmake script in the Module
        # directory.
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/Modules/patch_json.cmake
    )
endif()

#------------------------------------------------------------------------------

get_filename_component(
    JSON_INCLUDE_DIR
    "${EXTERNAL_PROJECTS_ROOT}/ext_json-prefix/src/ext_json/include"
    ABSOLUTE)
set(JSON_INCLUDE_DIR "${JSON_INCLUDE_DIR}" PARENT_SCOPE)
