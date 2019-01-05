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

set(EIGEN_GIT_TAG patched)
set(EIGEN_GIT_URL https://github.com/NervanaSystems/eigen)

#------------------------------------------------------------------------------
# Download Eigen
#------------------------------------------------------------------------------

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    GIT_REPOSITORY ${EIGEN_GIT_URL}
    GIT_TAG ${EIGEN_GIT_TAG}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
add_library(libeigen INTERFACE)
target_include_directories(libeigen SYSTEM INTERFACE ${SOURCE_DIR})
add_dependencies(libeigen ext_eigen)
