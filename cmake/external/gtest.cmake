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

# Enable FetchContent CMake module
include(FetchContent)

#------------------------------------------------------------------------------
# Download and install GoogleTest ...
#------------------------------------------------------------------------------

SET(GTEST_GIT_LABEL release-1.8.1)
#SET(GTEST_GIT_LABEL release-1.10.0)
SET(GTEST_ARCHIVE_URL https://github.com/google/googletest/archive/${GTEST_GIT_LABEL}.zip)
SET(GTEST_ARCHIVE_HASH 9ea36bf6dd6383beab405fd619bdce05e66a6535)

message(STATUS "Fetching googletest")

FetchContent_Declare(
    ext_gtest
    URL      ${GTEST_ARCHIVE_URL}
    #URL_HASH SHA1=${GTEST_ARCHIVE_HASH}
)

FetchContent_GetProperties(ext_gtest)
if(NOT ext_gtest_POPULATED)
    FetchContent_Populate(ext_gtest)
    add_subdirectory(${ext_gtest_SOURCE_DIR} ${ext_gtest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(libgtest INTERFACE)
target_link_libraries(libgtest INTERFACE gtest gmock)
