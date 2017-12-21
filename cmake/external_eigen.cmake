# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Enable ExternalProject CMake module
include(ExternalProject)

set(EIGEN_INSTALL_DIR ${EXTERNAL_INSTALL_DIR}/eigen)
set(EIGEN_PROJECT eigen)
set(EIGEN_GIT_TAG d608d9f3f577118981acbdd40da9dcf6b514668a)
set(EIGEN_GIT_URL https://github.com/jmenon/eigen)

#----------------------------------------------------------------------------------------------------------
# Download and install Eigen
#----------------------------------------------------------------------------------------------------------

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ${EIGEN_PROJECT}
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EIGEN_INSTALL_DIR}
        )
else()
    ExternalProject_Add(
        ${EIGEN_PROJECT}
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EIGEN_INSTALL_DIR}
        BUILD_BYPRODUCTS "${EIGEN_INSTALL_DIR}/include/eigen3"
        )
endif()

#----------------------------------------------------------------------------------------------------------

ExternalProject_Get_Property(eigen source_dir binary_dir)

set(EIGEN_INCLUDE_DIR "${EIGEN_INSTALL_DIR}/include/eigen3" PARENT_SCOPE)
