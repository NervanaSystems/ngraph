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

include(ExternalProject)

set(MLIR_LLVM_REPO_URL https://github.com/llvm/llvm-project.git)

# Change these commit IDs to move to latest stable versions
set(MLIR_LLVM_COMMIT_ID 376c6853)

# MLIR environment variables. Some of them are used by LIT tool.

if (NGRAPH_USE_PREBUILT_MLIR)
    set(MLIR_PROJECT_ROOT ${MLIR_LLVM_PREBUILT_PATH})
else()
    set(MLIR_PROJECT_ROOT ${CMAKE_CURRENT_BINARY_DIR}/mlir_project)
endif()

set(MLIR_LLVM_ROOT ${MLIR_PROJECT_ROOT}/llvm-project)
set(MLIR_LLVM_SOURCE_DIR ${MLIR_LLVM_ROOT}/llvm)
set(MLIR_SOURCE_DIR ${MLIR_LLVM_ROOT}/mlir)
set(MLIR_LLVM_BUILD_DIR ${MLIR_PROJECT_ROOT}/build)
set(MLIR_LLVM_TOOLS_DIR ${MLIR_LLVM_BUILD_DIR}/bin)
set(NGRAPH_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(NGRAPH_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/mlir)

# MLIR has to be pre-built before ngraph build starts
# this will clone and build MLIR during cmake config instead

# we will fetch and build it from the source if cmake is not configured to use
# the prebuilt mlir
if (NOT NGRAPH_USE_PREBUILT_MLIR)
    configure_file(${CMAKE_SOURCE_DIR}/cmake/mlir_fetch.cmake.in ${MLIR_PROJECT_ROOT}/CMakeLists.txt @ONLY)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                    -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
                    -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
                    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_ORIGINAL_CXX_FLAGS} .
                    WORKING_DIRECTORY "${MLIR_PROJECT_ROOT}")

    # Clone and build llvm + mlir.
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                    WORKING_DIRECTORY "${MLIR_PROJECT_ROOT}")
endif()

# Enable modules for LLVM.
set(LLVM_DIR "${MLIR_LLVM_BUILD_DIR}/lib/cmake/llvm"
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
set(MLIR_BIN_INCLUDE_PATH ${MLIR_LLVM_BUILD_DIR}/tools/mlir/include)
set(MLIR_INCLUDE_PATHS ${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH})
set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
