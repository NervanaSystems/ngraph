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

SET(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
SET(GTEST_GIT_LABEL release-1.10.0)

if(WIN32)
    list(APPEND GTEST_CMAKE_ARGS
        -Dgtest_force_shared_crt=TRUE
    )
endif()

if(UNIX)
    # workaround for compile error
    # related: https://github.com/intel/mkl-dnn/issues/55
    set(GTEST_CXX_FLAGS "-Wno-unused-result ${CMAKE_ORIGINAL_CXX_FLAGS} -Wno-undef")
else()
    set(GTEST_CXX_FLAGS ${CMAKE_ORIGINAL_CXX_FLAGS})
endif()

set(GTEST_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/gtest)

FetchContent_Declare(
    ext_gtest
    GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
    GIT_TAG        ${GTEST_GIT_LABEL}
    GIT_SHALLOW    1
)

FetchContent_GetProperties(ext_gtest)
if(NOT ext_gtest_POPULATED)
    FetchContent_Populate(ext_gtest)
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
    -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
    -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
    ${NGRAPH_FORWARD_CMAKE_ARGS}
    -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
    ${GTEST_CMAKE_ARGS}
    -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_PREFIX}
    ${ext_gtest_SOURCE_DIR}
    WORKING_DIRECTORY "${ext_gtest_BINARY_DIR}")

if("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
    include(ProcessorCount)
    ProcessorCount(N)
    if(N EQUAL 0)
        set(N 8)
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . --target install -- -j${N}
    WORKING_DIRECTORY "${ext_gtest_BINARY_DIR}")
elseif(NGRAPH_GENERATOR_IS_MULTI_CONFIG)
    foreach(BUILD_CONFIG ${CMAKE_CONFIGURATION_TYPES})
        execute_process(COMMAND "${CMAKE_COMMAND}" --build . --target install --config ${BUILD_CONFIG}
        WORKING_DIRECTORY "${ext_gtest_BINARY_DIR}")
    endforeach()
else()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . --target install
    WORKING_DIRECTORY "${ext_gtest_BINARY_DIR}")
endif()
#------------------------------------------------------------------------------

set(GTest_DIR ${GTEST_INSTALL_PREFIX}/lib/cmake/GTest)

find_package(GTest REQUIRED CONFIG)

add_library(libgtest INTERFACE)
target_link_libraries(libgtest INTERFACE GTest::gtest GTest::gmock)
