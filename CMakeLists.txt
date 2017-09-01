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

cmake_minimum_required (VERSION 2.8)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

project (ngraph)

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
    message("setting clang flags...")
    include( cmake/clang_4_0_flags.cmake )
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

