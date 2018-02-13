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
        gtest
        GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
        GIT_TAG ${GTEST_GIT_LABEL}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
        # Disable install step
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        )
else()
    ExternalProject_Add(
        gtest
        GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
        GIT_TAG ${GTEST_GIT_LABEL}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
        # Disable install step
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/gtest/src/gtest-build/googlemock/gtest/libgtest.a"
        )
endif()

#----------------------------------------------------------------------------------------------------------

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

get_filename_component(
    GTEST_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/gtest/src/gtest/googletest/include"
    ABSOLUTE)
set(GTEST_INCLUDE_DIR "${GTEST_INCLUDE_DIR}" PARENT_SCOPE)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
