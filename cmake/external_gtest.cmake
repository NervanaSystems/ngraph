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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install GoogleTest ...
#------------------------------------------------------------------------------

set(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
set(GTEST_GIT_LABEL release-1.8.1)
set(GTEST_PROJECT_ROOT ${EXTERNAL_PROJECTS_ROOT}/gtest-project)
set(GTEST_SOURCE_DIR ${GTEST_PROJECT_ROOT}/gtest-src)
set(GTEST_BINARY_DIR ${GTEST_PROJECT_ROOT}/gtest-build)

configure_file(${CMAKE_SOURCE_DIR}/cmake/gtest_fetch.cmake.in ${GTEST_PROJECT_ROOT}/CMakeLists.txt @ONLY)

execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
    -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
    -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
    WORKING_DIRECTORY "${GTEST_PROJECT_ROOT}")

# clone and build googletest
include(ProcessorCount)
ProcessorCount(N)
if(("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles") AND (NOT N EQUAL 0))
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${N}
        WORKING_DIRECTORY "${GTEST_PROJECT_ROOT}")
else()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${GTEST_PROJECT_ROOT}")
endif()

add_subdirectory("${GTEST_SOURCE_DIR}"
                 "${GTEST_BINARY_DIR}"
)

if(WIN32)
    target_compile_definitions(gtest PRIVATE gtest_force_shared_crt=ON)
    target_compile_definitions(gmock PRIVATE gtest_force_shared_crt=ON)
endif()

if(UNIX)
    # workaround for compile error
    # related: https://github.com/intel/mkl-dnn/issues/55
    target_compile_options(gtest PRIVATE -Wno-unused-result -Wno-undef)
    target_compile_options(gmock PRIVATE -Wno-unused-result -Wno-undef)
endif()

