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

# TODO: File is getting to big. Refactor
if(NGRAPH_USE_PREBUILT_LLVM)
    message(STATUS "LLVM: Using prebuilt LLVM")
    # Override default LLVM binaries
    if(NOT DEFINED LLVM_TARBALL_URL)
        set(NGRAPH_LLVM_VERSION 9.0.1)
        if(APPLE)
            set(LLVM_TARBALL_URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${NGRAPH_LLVM_VERSION}/clang+llvm-${NGRAPH_LLVM_VERSION}-x86_64-apple-darwin.tar.xz)
            set(LLVM_SHA1_HASH 693e7c2d2fcd005f0d8198ea0174ae30ec78bb7c)
        elseif(UNIX AND NOT APPLE)
            if(EXISTS /etc/lsb-release)
                execute_process(COMMAND grep DISTRIB_RELEASE /etc/lsb-release OUTPUT_VARIABLE UBUNTU_VER_LINE)
                string(REGEX MATCH "[0-9.]+" UBUNTU_VERSION ${UBUNTU_VER_LINE})
                message(STATUS "Ubuntu version: ${UBUNTU_VERSION} detected.")
                if(UBUNTU_VERSION VERSION_LESS 16.04)
                    message(FATAL_ERROR "No prebuilt LLVM available for Ubuntu ${UBUNTU_VER} on llvm.org, please set LLVM_TARBALL_URL manually.")
                else()
                    set(LLVM_SHA1_HASH 608285fb0503ee6d15ce9acb8f4575bba4d78e06)
                endif()
                set(LLVM_TARBALL_URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${NGRAPH_LLVM_VERSION}/clang+llvm-${NGRAPH_LLVM_VERSION}-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
            else()
                message(FATAL_ERROR "Prebuilt LLVM: Only Ubuntu Linux is supported.")
            endif()
        else()
            message(FATAL_ERROR "Prebuilt LLVM: unsupported OS.")
        endif()
    else()
        if(NOT DEFINED LLVM_SHA1_HASH)
            message(FATAL_ERROR "Prebuilt LLVM: please provide LLVM_SHA_HASH.")
        endif()
    endif()

    FetchContent_Declare(
        prebuilt_llvm
        URL ${LLVM_TARBALL_URL}
        URL_HASH SHA1=${LLVM_SHA1_HASH}
        SOURCE_DIR ${LLVM_ROOT}
        )

    FetchContent_GetProperties(prebuilt_llvm)
    if(NOT prebuilt_llvm_POPULATED)
        FetchContent_Populate(prebuilt_llvm)
    endif()
endif()

# Try to find system or user provide Clang first and use it if available
# Clang Config does not support version so find LLVM first
# To install Clang 9 system wide On Ubuntu 18.04
# sudo apt-get install clang-9 libclang-9-dev
# For non-system clang, provide LLVM_ROOT by passing
# -DLLVM_ROOT=<CMAKE_INSTALL_PREFIX that was used for build or top level directory of unpacked LLVM release from github>
# When you configure CMake
if(NGRAPH_CODEGEN_ENABLE)
    find_package(LLVM 9 CONFIG)
    if(LLVM_FOUND)
        find_package(Clang CONFIG
            HINTS ${LLVM_DIR}/../lib/cmake/clang ${LLVM_DIR}/../clang NO_DEFAULT_PATH)
        if(Clang_FOUND)
            set(NEED_TO_BUILD_LLVM FALSE)
        endif()
    endif()
elseif(NGRAPH_MLIR_ENABLE)
    if(NGRAPH_USE_PREBUILT_MLIR)
        set(LLVM_ROOT ${MLIR_LLVM_PREBUILT_PATH})
    endif()
    set(MLIR_COMMIT_ID 3c5dd5863c34ecd51e9d2a49929877d8151dea39)
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
endif()

if(NEED_TO_BUILD_LLVM)
    if(NOT NGRAPH_OVERWRITE_LLVM_ROOT)
        message(FATAL_ERROR "nGraph is not allowed overwrite contents at LLVM_ROOT: ${LLVM_ROOT} "
            "Set NGRAPH_OVERWRITE_LLVM_ROOT to ON if you would like to overwrite.")
    endif()
    message(STATUS "LLVM: Building LLVM from source")

    set(LLVM_GIT_REPOSITORY https://github.com/llvm/llvm-project.git)
    if(NGRAPH_CPU_CODEGEN_ENABLE)
        set(LLVM_GIT_TAG llvmorg-9.0.1)
        set(LLVM_CMAKE_ARGS
            -DLLVM_ENABLE_PROJECTS:STRING=clang\;openmp
            -DLLVM_INCLUDE_DOCS=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DLLVM_INCLUDE_GO_TESTS=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_BENCHMARKS=OFF
            -DLLVM_BUILD_TOOLS=OFF
            -DLLVM_BUILD_UTILS=OFF
            -DLLVM_BUILD_RUNTIMES=OFF
            -DLLVM_BUILD_RUNTIME=OFF
            -DLLVM_TARGETS_TO_BUILD=host
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_ENABLE_TERMINFO=OFF
            -DLLVM_ENABLE_ZLIB=OFF
            -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON
            -DLLVM_ENABLE_WARNINGS=OFF
            -DLLVM_ENABLE_PEDANTIC=OFF
            -DLIBOMP_OMPT_SUPPORT=OFF
            -DCLANG_ENABLE_ARCMT=OFF
            -DCLANG_ENABLE_STATIC_ANALYZER=OFF)
        set(LLVM_GIT_SHALLOW 1)
    elseif(NGRAPH_MLIR_ENABLE)
        set(LLVM_GIT_TAG ${MLIR_COMMIT_ID})
        set(LLVM_CMAKE_ARGS
            -DLLVM_ENABLE_RTTI=ON
            -DLLVM_ENABLE_PROJECTS:STRING=mlir
            -DLLVM_BUILD_EXAMPLES=ON
            -DLLVM_TARGETS_TO_BUILD=host)
        set(LLVM_GIT_SHALLOW 0)
        set(LLVM_PATCH_COMMAND git reset HEAD --hard COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/mlir.patch)
    endif()


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

    if(NGRAPH_MLIR_ENABLE)
        # MLIR needs build tree
        set(llvm_BINARY_DIR ${LLVM_ROOT})
        message(STATUS "Override llvm_BINARY_DIR: ${llvm_BINARY_DIR}")
        if(NOT EXISTS ${LLVM_ROOT})
            file(MAKE_DIRECTORY ${LLVM_ROOT})
        endif()
    endif()

    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
        -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}
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
    # MLIR needs build tree
    if(NGRAPH_CODEGEN_ENABLE)
        execute_process(COMMAND "${CMAKE_COMMAND}" --install .
            WORKING_DIRECTORY "${llvm_BINARY_DIR}")
    endif()
endif()

if(NGRAPH_CODEGEN_ENABLE)
    if(NOT Clang_FOUND)
        find_package(Clang REQUIRED CONFIG
            HINTS ${LLVM_ROOT}/lib/cmake/clang NO_DEFAULT_PATH)
    endif()
    message(STATUS "CLANG_CMAKE_DIR: ${CLANG_CMAKE_DIR}")
    message(STATUS "CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

    add_library(libllvm INTERFACE)
    target_include_directories(libllvm INTERFACE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIR})
    target_link_libraries(libllvm INTERFACE clangHandleCXX clangHandleLLVM)
elseif(NGRAPH_MLIR_ENABLE)
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

    # Enable modules for LLVM.
    set(LLVM_DIR "${LLVM_ROOT}/lib/cmake/llvm"
        CACHE PATH "Path to LLVM cmake modules")
    message(STATUS "LLVM DIR: ${LLVM_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
    include(AddLLVM)

    # Enable LLVM package, definitions and env vars.
    find_package(LLVM REQUIRED CONFIG
        HINTS ${LLVM_ROOT} NO_DEFAULT_PATH)
    add_definitions(${LLVM_DEFINITIONS})
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using modules in: ${LLVM_DIR}")
    message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

    set(MLIR_DIR "${LLVM_ROOT}/lib/cmake/mlir"
        CACHE PATH "Path to MLIR cmake modules")
    list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
    find_package(MLIR REQUIRED CONFIG)

    # Used by tblgen
    set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
    set(MLIR_BIN_INCLUDE_PATH ${MLIR_INCLUDE_DIR})
    # Used by ngraph mlir and cpu backend
    set(MLIR_INCLUDE_PATHS ${MLIR_INCLUDE_DIRS})
    set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

    message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
    message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
endif()
