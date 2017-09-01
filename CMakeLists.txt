# Copyright 2017 Nervana Systems Inc.
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

# We require version 3.5.1 only because that's the lowest version of CMake on which
# we've validate this repository.
cmake_minimum_required (VERSION 3.5.1)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

if ("${CMAKE_CXX_COMPILER}" STREQUAL "")
    # The user didn't indicate a compiler preference, to pick the compiler based on
    # the personal preferences and convenience of the project's developers...
    find_program(
        CXX_COMPILER
        NAMES
        clang++-4.0
        clang++-3.9
        clang++
        c++
        NAMES_PER_DIR
        DOC "Which C++ compiler to use when building `libngraph` and the tests."
        )

    if ("${CXX_COMPILER}" STREQUAL "CXX_COMPILER-NOTFOUND")
        message(FATAL_ERROR "Unable to find a suitable c++ compiler.")
    else()
        message(STATUS "Using C++ compiler: '${CXX_COMPILER}'")
        set(CMAKE_CXX_COMPILER "${CXX_COMPILER}")
    endif()
endif()

project (ngraph
    LANGUAGES CXX
    VERSION 0.1
    )

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------

# Default values...
set(NGRAPH_CXX_WARNING_FLAGS "")

# Compiler-specific logic...
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    message(STATUS "Using project warning flags")
    include( cmake/clang_4_0_flags.cmake )
else()
    message(WARNING "CMAKE_CXX_COMPILER_ID='${CMAKE_CXX_COMPILER_ID}'.  CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}'.  The only supported compiler is 'clang', but the selected compiler is '${CMAKE_CXX_COMPILER}'.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

#-----------------------------------------------------------------------------------------------

# The following 'add_subdirectory' call:
# - defines the 'libgtest' target.
# - defines the 'GTEST_INCLUDE_DIR' cmake variable.
add_subdirectory(third-party)

# The following 'add_subdirectory' call:
# - defines the 'ngraph' library target.
# - defines the 'NGRAPH_INCLUDE_DIR' cmake variable.
# - defines the 'NGRAPH_VERSION' cmake variable.
# - install-related targets.
add_subdirectory(src)

add_subdirectory(test)

