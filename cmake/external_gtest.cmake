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
# Download and install GoogleTest ...
#------------------------------------------------------------------------------

SET(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
SET(GTEST_GIT_LABEL release-1.8.1)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    endif()
endif()

set(GTEST_OUTPUT_DIR ${EXTERNAL_PROJECTS_ROOT}/gtest/build/googlemock/gtest)

if (APPLE OR LINUX)
    set(COMPILE_FLAGS -fPIC)
endif()

set(GTEST_CMAKE_ARGS
    -DCMAKE_CXX_FLAGS=${COMPILE_FLAGS}
)
if(WIN32)
    list(APPEND GTEST_CMAKE_ARGS
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE=${GTEST_OUTPUT_DIR}
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG=${GTEST_OUTPUT_DIR}
        -Dgtest_force_shared_crt=TRUE
    )
endif()

ExternalProject_Add(
    ext_gtest
    PREFIX gtest
    GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
    GIT_TAG ${GTEST_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        ${GTEST_CMAKE_ARGS}
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/gtest/build"
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

add_library(libgtest INTERFACE)
add_dependencies(libgtest ext_gtest)
target_include_directories(libgtest SYSTEM INTERFACE ${SOURCE_DIR}/googletest/include)

target_link_libraries(libgtest INTERFACE
    debug ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtestd${CMAKE_STATIC_LIBRARY_SUFFIX}
    optimized ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
