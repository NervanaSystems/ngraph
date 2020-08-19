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

set(NGRAPH_EXTERNAL_LLVM_BUILD_DIR "" CACHE STRING "Path to prebuilt LLVM build tree.")

if(NGRAPH_USE_PREBUILT_MLIR)
    set(NGRAPH_EXTERNAL_LLVM_BUILD_DIR ${MLIR_LLVM_PREBUILT_PATH}/build)
endif()

if(NOT ("${NGRAPH_EXTERNAL_LLVM_BUILD_DIR}" STREQUAL ""))
    set(TRY_EXTERNAL_LLVM_BUILD TRUE)
endif()

set(LLVM_COMMIT_ID f48eced390dcda54766e1c510af10bbcbaebcd7e)
if(TRY_EXTERNAL_LLVM_BUILD)
    set(VCSREVISION "${NGRAPH_EXTERNAL_LLVM_BUILD_DIR}/include/llvm/Support/VCSRevision.h")
    if(EXISTS "${VCSREVISION}")
        message(STATUS "LLVM: VCSRevision.h found.")
        file(READ "${VCSREVISION}" REVISION_FILE)
        if(${REVISION_FILE} MATCHES "^#undef LLVM_REVISION*")
            # LLVM is built from a source archive
            message(WARNING "LLVM: Could not revision. Make sure commit ID is ${LLVM_COMMIT_ID}")
            set(USE_EXTERNAL_LLVM_BUILD TRUE)
        else()
            string(REGEX MATCH "LLVM_REVISION \"([A-Za-z0-9]+)\"" _ ${REVISION_FILE})
            set(LONG_REV ${CMAKE_MATCH_1})
            if(LONG_REV STREQUAL LLVM_COMMIT_ID)
                message(STATUS "LLVM: Revision Matches.")
                set(USE_EXTERNAL_LLVM_BUILD TRUE)
            endif()
        endif()
    endif()
endif()

if(USE_EXTERNAL_LLVM_BUILD)
    if(NGRAPH_CODEGEN_ENABLE)
        find_package(Clang REQUIRED CONFIG HINTS "${NGRAPH_EXTERNAL_LLVM_BUILD_DIR}/lib/cmake/clang" NO_DEFAULT_PATH)
    endif()
    if(NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE)
        find_package(MLIR REQUIRED CONFIG HINTS "${NGRAPH_EXTERNAL_LLVM_BUILD_DIR}/lib/cmake/mlir" NO_DEFAULT_PATH)
        include(${NGRAPH_EXTERNAL_LLVM_BUILD_DIR}/lib/cmake/llvm/TableGen.cmake)
        # Enable LLVM package, definitions and env vars.
        add_definitions(${LLVM_DEFINITIONS})
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")
    endif()
    set(llvm_BINARY_DIR ${NGRAPH_EXTERNAL_LLVM_BUILD_DIR})
    set(llvm_SOURCE_DIR ${llvm_BINARY_DIR}/..)
else()
    if (NGRAPH_USE_PREBUILT_LLVM)
        message(FATAL_ERROR "LLVM: prebuilt is the wrong hash, expected ${LLVM_COMMIT_ID}")
    endif()
    message(STATUS "LLVM: Building LLVM from source")
    message(STATUS "LLVM: Fetching source")

    set(LLVM_ARCHIVE_URL https://github.com/llvm/llvm-project/archive/${LLVM_COMMIT_ID}.zip)
    set(LLVM_ARCHIVE_URL_HASH b21201e6a8c59fb5ca4f9ae9190f044996d40321)

    FetchContent_Declare(
        llvm
        URL ${LLVM_ARCHIVE_URL}
        URL_HASH SHA1=${LLVM_ARCHIVE_URL_HASH}
        )

    set(LLVM_ENABLE_RTTI ON CACHE INTERNAL "")
    set(LLVM_TARGETS_TO_BUILD host CACHE INTERNAL "")
    if(NGRAPH_CODEGEN_ENABLE)
        set(LLVM_ENABLE_PROJECTS "clang;openmp;mlir" CACHE INTERNAL "")
    else()
        set(LLVM_ENABLE_PROJECTS "mlir" CACHE INTERNAL "")
    endif()

    FetchContent_GetProperties(llvm)
    if(NOT llvm_POPULATED)
        FetchContent_Populate(llvm)
        add_subdirectory(${llvm_SOURCE_DIR}/llvm ${llvm_BINARY_DIR})
    endif()

    # In subdirectory build cannot use cmake config file and need to set some variables manually
    set(LLVM_CMAKE_DIR ${llvm_SOURCE_DIR}/llvm/cmake/modules)
    set(LLVM_INCLUDE_DIRS ${llvm_SOURCE_DIR}/llvm/include ${llvm_BINARY_DIR}/include)
    if(NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE)
        set(MLIR_CMAKE_DIR ${llvm_SOURCE_DIR}/mlir/cmake/modules)
        set(MLIR_INCLUDE_DIR ${llvm_BINARY_DIR}/tools/mlir/include)
        set(MLIR_INCLUDE_DIRS ${llvm_SOURCE_DIR}/mlir/include ${llvm_BINARY_DIR}/tools/mlir/include)
        set(MLIR_TABLEGEN_EXE "mlir-tblgen")
    endif()
    if(NGRAPH_CODEGEN_ENABLE)
        set(CLANG_INCLUDE_DIRS ${llvm_SOURCE_DIR}/clang/include ${llvm_BINARY_DIR}/tools/clang/include)
        set(LLVM_INCLUDE_DIR ${llvm_BINARY_DIR}/include)
        set(LLVM_VERSION_MAJOR 12)
        set(LLVM_VERSION_MINOR 0)
        set(LLVM_VERSION_PATCH 0)
    endif()
endif()

if(NGRAPH_CODEGEN_ENABLE)
    message(STATUS "CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
    add_library(libllvm INTERFACE)
    target_include_directories(libllvm INTERFACE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
    target_link_libraries(libllvm INTERFACE clangHandleCXX clangHandleLLVM)
endif()

if(NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE)
    # MLIR environment variables. Some of them are used by LIT tool.

    # Only used in this file
    set(MLIR_SOURCE_DIR ${llvm_SOURCE_DIR}/mlir)
    # Used in test/mlir:
    # lit cfg
    set(MLIR_LLVM_BUILD_DIR ${llvm_BINARY_DIR})
    set(NGRAPH_LIT_TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test/mlir)
    set(NGRAPH_LIT_TEST_BUILD_DIR ${PROJECT_BINARY_DIR}/test/mlir)
    # lit cfg and path to llvm-lit
    set(MLIR_LLVM_TOOLS_DIR ${MLIR_LLVM_BUILD_DIR}/bin)

    # Enable modules for LLVM.
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    message(STATUS "Using LLVM modules in: ${LLVM_CMAKE_DIR}")
    include(AddLLVM)

    # Enable modules for MLIR.
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    message(STATUS "Using MLIR modules in: ${MLIR_CMAKE_DIR}")
    include(AddMLIR)

    # Used by tblgen
    set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
    set(MLIR_BIN_INCLUDE_PATH ${MLIR_INCLUDE_DIR})
    # Used by ngraph mlir and cpu backend
    set(MLIR_INCLUDE_PATHS ${MLIR_INCLUDE_DIRS})
    set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

    message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
    message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
endif()
