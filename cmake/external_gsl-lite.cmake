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
# Download and install gsl-lite...
#------------------------------------------------------------------------------

set(GSL_LITE_GIT_REPO_URL https://github.com/martinmoene/gsl-lite.git)
set(GSL_LITE_GIT_SHA1 119af9fcd3138cf0b7ee8f556cef884814b46095)

if (${CMAKE_VERSION} VERSION_LESS 3.6)
    ExternalProject_Add(
            ext_gsl-lite
            PREFIX ext_gsl-lite
            GIT_REPOSITORY ${GSL_LITE_GIT_REPO_URL}
            GIT_TAG ${GSL_LITE_GIT_SHA1}
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/tmp"
            STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/stamp"
            DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/download"
            SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/src"
            BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/bin"
            INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite"
            EXCLUDE_FROM_ALL TRUE)
else()
    # To speed things up prefer 'shallow copy' for CMake 3.6 and later
    ExternalProject_Add(
            ext_gsl-lite
            PREFIX ext_gsl-lite
            GIT_REPOSITORY ${GSL_LITE_GIT_REPO_URL}
            GIT_TAG ${GSL_LITE_GIT_SHA1}
            GIT_SHALLOW TRUE
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/tmp"
            STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/stamp"
            DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/download"
            SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/src"
            BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite/bin"
            INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/gsl-lite"
            EXCLUDE_FROM_ALL TRUE)
endif()

# -----------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gsl-lite SOURCE_DIR)

set(GSL_LITE_INCLUDE_DIR ${SOURCE_DIR}/include
    CACHE INTERNAL "Include directory for gsl-lite")

add_library(gsl INTERFACE)
target_include_directories(gsl SYSTEM INTERFACE ${GSL_LITE_INCLUDE_DIR})
add_dependencies(gsl ext_gsl-lite)
