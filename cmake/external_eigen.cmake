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

set(EIGEN_GIT_TAG patched)
set(EIGEN_GIT_URL https://github.com/NervanaSystems/eigen)

#----------------------------------------------------------------------------------------------------------
# Download Eigen
#----------------------------------------------------------------------------------------------------------

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_eigen
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen"
        )
else()
    ExternalProject_Add(
        ext_eigen
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen"
        BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/eigen/src/Eigen/Core"
        )
endif()

#----------------------------------------------------------------------------------------------------------

set(EIGEN_INCLUDE_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/src" PARENT_SCOPE)
