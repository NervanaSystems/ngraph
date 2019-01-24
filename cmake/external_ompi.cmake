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
# Download and install OpenMPI.
#------------------------------------------------------------------------------

set(OMPI_GIT_REPO_URL https://github.com/open-mpi/ompi.git)
set(OMPI_GIT_BRANCH v2.1.1)

# The 'BUILD_BYPRODUCTS' arguments was introduced in CMake 3.2.

ExternalProject_Add(
    ext_protobuf
    PREFIX protobuf
    GIT_REPOSITORY ${OMPI_GIT_REPO_URL}
    GIT_TAG ${OMPI_GIT_BRANCH}
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
