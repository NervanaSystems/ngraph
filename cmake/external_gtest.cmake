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

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG     release-1.8.0
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/gtest/src/gtest-build/googlemock/gtest/libgtest.a"
    )

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
