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

ExternalProject_Add(
    ext_mlir
    PREFIX llvm
    URL ${MLIR_LLVM_TARBALL_URL}
    URL_HASH SHA1=${MLIR_LLVM_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_mlir SOURCE_DIR)
set(INSTALL_DIR ${SOURCE_DIR})

# MLIR environment variables. Some of them are used by LIT tool.
set(MLIR_LLVM_ROOT ${INSTALL_DIR}/llvm-projects)
set(MLIR_SOURCE_DIR ${MLIR_LLVM_ROOT}/llvm/projects/mlir)
set(MLIR_BUILD_DIR ${MLIR_LLVM_ROOT}/build)
set(MLIR_TOOLS_DIR ${MLIR_BUILD_DIR}/bin)
set(NGRAPH_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(NGRAPH_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/mlir)

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
