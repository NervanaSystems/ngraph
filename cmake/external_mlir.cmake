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

include(ExternalProject)

set(MLIR_LLVM_REPO_URL https://github.com/llvm/llvm-project.git)
set(MLIR_REPO_URL https://github.com/tensorflow/mlir.git)

# Change these commit IDs to move to latest stable versions
set(MLIR_LLVM_COMMIT_ID 0845ac7331e)
set(MLIR_COMMIT_ID 1f7893e0)

# MLIR environment variables. Some of them are used by LIT tool.
set(MLIR_PROJECT_ROOT ${CMAKE_CURRENT_BINARY_DIR}/mlir_project)
set(MLIR_LLVM_ROOT ${MLIR_PROJECT_ROOT}/llvm-projects)
set(MLIR_SOURCE_DIR ${MLIR_LLVM_ROOT}/llvm/projects/mlir)
set(MLIR_BUILD_DIR ${MLIR_LLVM_ROOT}/build)
set(MLIR_TOOLS_DIR ${MLIR_BUILD_DIR}/bin)
set(NGRAPH_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(NGRAPH_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/mlir)

# MLIR has to be pre-built before ngraph build starts
# this will clone and build MLIR during cmake config instead

configure_file(${CMAKE_SOURCE_DIR}/cmake/mlir_fetch.cmake.in ${MLIR_PROJECT_ROOT}/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${MLIR_PROJECT_ROOT}")

# clone and build llvm
execute_process(COMMAND "${CMAKE_COMMAND}" --build . --target ext_mlir_llvm
                WORKING_DIRECTORY "${MLIR_PROJECT_ROOT}")

# clone and build mlir
execute_process(COMMAND "${CMAKE_COMMAND}" --build . --target ext_mlir
                WORKING_DIRECTORY "${MLIR_PROJECT_ROOT}")

# Enable modules for LLVM.
set(LLVM_DIR "${MLIR_BUILD_DIR}/lib/cmake/llvm"
    CACHE PATH "Path to LLVM cmake modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
include(AddLLVM)

# Enable LLVM package, definitions and env vars.
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using modules in: ${LLVM_DIR}")
message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
set(MLIR_BIN_INCLUDE_PATH ${MLIR_BUILD_DIR}/projects/mlir/include)
set(MLIR_INCLUDE_PATHS ${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH})
set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
