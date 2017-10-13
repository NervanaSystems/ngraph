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

# link_directories(), not recommended
# include_directories() for headers

# Enable ExternalProject CMake module
include(ExternalProject)

#-----------------------------------------------------------------------------------------------
# External Argon dependencies
#-----------------------------------------------------------------------------------------------
# By default, Argon libraries is at $NRV_SYSTEM_GIT_ROOT/_out/$BUILD_PLATFORM/lib, which should be
# added to LD_LIBRARY_PATH already during Argon setup.
message(STATUS "Lib path: $ENV{NRV_SYSTEM_GIT_ROOT}/_out/$ENV{BUILD_PLATFORM}/lib")
find_library(ARGON_EXTERNAL_LIB
    NAMES
    argon_api
    optimizer
    ir_builder
    umd
    flex
    disasm
    HINTS
    $ENV{NRV_SYSTEM_GIT_ROOT}/_out/$ENV{BUILD_PLATFORM}/lib
    $ENV{LD_LIBRARY_PATH}
)

message(STATUS "Header path: $ENV{NRV_SYSTEM_GIT_ROOT}/_out/$ENV{BUILD_PLATFORM}/include")
# Argon headers
include_directories(
    $ENV{NRV_SYSTEM_GIT_ROOT}/_out/$ENV{BUILD_PLATFORM}/include
)

#-----------------------------------------------------------------------------------------------
# Argon backend unit tests
#-----------------------------------------------------------------------------------------------
# Sources
set (ARGON_BACKEND_UNIT_TEST_SRC
    tests/main.cpp
    tests/simple_ew.cpp
)

# Executable
add_executable(argon_backend_unit_test ${ARGON_BACKEND_UNIT_TEST_SRC})

# Link libs
target_link_libraries(argon_backend_unit_test ${ARGON_EXTERNAL_LIB})
target_link_libraries(argon_backend_unit_test ngraph libgtest pthread)
target_link_libraries(argon_backend_unit_test ${CMAKE_DL_LIBS})

# # MKL Lib and Dependencies
# if(MKLDNN_INCLUDE_DIR)
#     target_link_libraries(argon_backend_unit_test mkldnn)
#     add_dependencies(argon_backend_unit_test ext_mkldnn)
# endif()

# Headers
include_directories(
    SYSTEM
    ${GTEST_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)
