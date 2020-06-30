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

include(FetchContent)
set(LLVM_ROOT ${EXTERNAL_PROJECTS_ROOT}/llvm CACHE STRING "Path to LLVM installation.")
if("${LLVM_ROOT}" STREQUAL "${EXTERNAL_PROJECTS_ROOT}/llvm")
    set(NGRAPH_OVERWRITE_LLVM_ROOT ON CACHE BOOL "Overwrite contents in LLVM_ROOT if version does not match.")
else()
    set(NGRAPH_OVERWRITE_LLVM_ROOT OFF CACHE BOOL "Overwrite contents in LLVM_ROOT if version does not match.")
endif()

set(NEED_TO_BUILD_LLVM TRUE)

# Try to find system or user provide Clang first and use it if available
# Clang Config does not support version so find LLVM first
# For non-system clang, provide LLVM_ROOT by passing
# -DLLVM_ROOT=<CMAKE_INSTALL_PREFIX that was used for build or top level directory of unpacked LLVM release from github>
# When you configure CMake
if(NGRAPH_USE_PREBUILT_MLIR)
    set(LLVM_ROOT ${MLIR_LLVM_PREBUILT_PATH}/build)
endif()
# TODO: remove this file after CI is updated.
include(cmake/external_mlir.cmake)
set(MLIR_COMMIT_ID ${MLIR_LLVM_COMMIT_ID})
set(VCSREVISION "${LLVM_ROOT}/include/llvm/Support/VCSRevision.h")
if(EXISTS "${VCSREVISION}")
    message(STATUS "LLVM_Revision found.")
    file(READ "${VCSREVISION}" REVISION_FILE)
    string(REGEX MATCH "LLVM_REVISION \"([A-Za-z0-9]+)\"" _ ${REVISION_FILE})
    set(LONG_REV ${CMAKE_MATCH_1})
    string(TOLOWER ${LONG_REV} LONG_REV)
    string(TOLOWER ${MLIR_COMMIT_ID} MLIR_COMMIT_ID)
    if(LONG_REV STREQUAL MLIR_COMMIT_ID)
        message(STATUS "SHA1 HASH Matches.")
        set(NEED_TO_BUILD_LLVM FALSE)
    endif()
endif()

if(NEED_TO_BUILD_LLVM)
    if(NOT NGRAPH_OVERWRITE_LLVM_ROOT)
        message(FATAL_ERROR "nGraph is not allowed overwrite contents at LLVM_ROOT: ${LLVM_ROOT} "
            "Set NGRAPH_OVERWRITE_LLVM_ROOT to ON if you would like to overwrite.")
    endif()
    message(STATUS "LLVM: Building LLVM from source")

    set(LLVM_GIT_REPOSITORY https://github.com/llvm/llvm-project.git)
        set(LLVM_GIT_TAG ${MLIR_COMMIT_ID})
        set(LLVM_GIT_SHALLOW 0)
        set(LLVM_PATCH_COMMAND git reset HEAD --hard COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/mlir.patch)

    FetchContent_Declare(
        llvm
        GIT_REPOSITORY ${LLVM_GIT_REPOSITORY}
        GIT_TAG ${LLVM_GIT_TAG}
        GIT_SHALLOW ${LLVM_GIT_SHALLOW}
        PATCH_COMMAND ${LLVM_PATCH_COMMAND}
        )

    FetchContent_GetProperties(llvm)
    if(NOT llvm_POPULATED)
        FetchContent_Populate(llvm)
    endif()

    # MLIR needs build tree
    set(llvm_BINARY_DIR ${LLVM_ROOT})
    message(STATUS "Override llvm_BINARY_DIR: ${llvm_BINARY_DIR}")
    if(NOT EXISTS ${LLVM_ROOT})
        file(MAKE_DIRECTORY ${LLVM_ROOT})
    endif()

    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
        -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}
        -DLLVM_ENABLE_PROJECTS:STRING=clang\;openmp\;mlir
            -DLLVM_ENABLE_RTTI=ON
            -DLLVM_ENABLE_TERMINFO=OFF
            -DLLVM_ENABLE_ZLIB=OFF
            -DLLVM_TARGETS_TO_BUILD=host
        ${LLVM_CMAKE_ARGS}
        ${llvm_SOURCE_DIR}/llvm
        WORKING_DIRECTORY "${llvm_BINARY_DIR}")

    # clone and build llvm
    include(ProcessorCount)
    ProcessorCount(N)
    if(("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles") AND (NOT N EQUAL 0))
        execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${N}
            WORKING_DIRECTORY "${llvm_BINARY_DIR}")
    else()
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${llvm_BINARY_DIR}")
    endif()
endif()

find_package(Clang REQUIRED CONFIG HINTS "${LLVM_ROOT}/lib/cmake/clang" NO_DEFAULT_PATH)
message(STATUS "CLANG_CMAKE_DIR: ${CLANG_CMAKE_DIR}")
message(STATUS "CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

add_library(libllvm INTERFACE)
target_include_directories(libllvm INTERFACE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(libllvm INTERFACE clangHandleCXX clangHandleLLVM)
# MLIR environment variables. Some of them are used by LIT tool.

# Only used in this file
set(MLIR_LLVM_ROOT ${llvm_SOURCE_DIR})
set(MLIR_LLVM_SOURCE_DIR ${MLIR_LLVM_ROOT}/llvm)
set(MLIR_SOURCE_DIR ${MLIR_LLVM_ROOT}/mlir)
# Used in test/mlir:
# lit cfg
set(MLIR_LLVM_BUILD_DIR ${LLVM_ROOT})
set(NGRAPH_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(NGRAPH_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/mlir)
# lit cfg and path to llvm-lit
set(MLIR_LLVM_TOOLS_DIR ${MLIR_LLVM_BUILD_DIR}/bin)

set(MLIR_ROOT ${LLVM_ROOT})
find_package(MLIR REQUIRED CONFIG HINTS "${LLVM_ROOT}/lib/cmake/mlir" NO_DEFAULT_PATH)

# Enable LLVM package, definitions and env vars.
add_definitions(${LLVM_DEFINITIONS})
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

# Enable modules for LLVM.
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "Using modules in: ${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Enable modules for MLIR.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
message(STATUS "Using modules in: ${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Used by tblgen
set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
set(MLIR_BIN_INCLUDE_PATH ${MLIR_INCLUDE_DIR})
# Used by ngraph mlir and cpu backend
set(MLIR_INCLUDE_PATHS ${MLIR_INCLUDE_DIRS})
set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
