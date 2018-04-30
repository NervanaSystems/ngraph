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

#----------------------------------------------------------------------------------------------------------
# Download and install GoogleTest ...
#----------------------------------------------------------------------------------------------------------

SET(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
SET(GTEST_GIT_LABEL release-1.8.0)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_gtest
        PREFIX gtest
        GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
        GIT_TAG ${GTEST_GIT_LABEL}
        # Disable install step
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_FLAGS="-fPIC"
        )
else()
    ExternalProject_Add(
        ext_gtest
        PREFIX gtest
        GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
        GIT_TAG ${GTEST_GIT_LABEL}
        # Disable install step
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_FLAGS="-fPIC"
        BUILD_BYPRODUCTS "gtest/ext_gtest-build/googlemock/gtest/libgtest.a"
        )
endif()

#----------------------------------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

set(GTEST_INCLUDE_DIR ${SOURCE_DIR}/googletest/include)
link_directories(${BINARY_DIR}/googlemock/gtest)
