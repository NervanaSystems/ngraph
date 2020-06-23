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

set(LLVM_ROOT ${EXTERNAL_PROJECTS_ROOT}/llvm CACHE STRING "Path to LLVM installation.")
if("${LLVM_ROOT}" STREQUAL "${EXTERNAL_PROJECTS_ROOT}/llvm")
    set(NGRAPH_OVERWRITE_LLVM_ROOT_ENABLE ON CACHE BOOL "Overwrite contents in LLVM_ROOT if version does not match.")
else()
    set(NGRAPH_OVERWRITE_LLVM_ROOT_ENABLE OFF CACHE BOOL "Overwrite contents in LLVM_ROOT if version does not match.")
endif()

# Try to find system or user provide Clang first and use it if available
# Clang Config does not support version so find LLVM first
# To install Clang 9 system wide On Ubuntu 18.04
# sudo apt-get install clang-9 libclang-9-dev
# For non-system clang, provide LLVM_ROOT by passing
# -DLLVM_ROOT=<CMAKE_INSTALL_PREFIX that was used for build or top level directory of unpacked LLVM release from github>
# When you configure CMake
set(NEED_TO_BUILD_LLVM TRUE)
find_package(LLVM 9 CONFIG)
if(LLVM_FOUND)
    find_package(Clang CONFIG
        HINTS ${LLVM_DIR}/../lib/cmake/clang ${LLVM_DIR}/../clang NO_DEFAULT_PATH)
    if(Clang_FOUND)
        set(NEED_TO_BUILD_LLVM FALSE)
    endif()
endif()

if(NEED_TO_BUILD_LLVM)
    if(NOT NGRAPH_OVERWRITE_LLVM_ROOT_ENABLE)
        message(FATAL_ERROR "nGraph is not allowed overwrite contents at LLVM_ROOT: ${LLVM_ROOT} "
            "Set NGRAPH_OVERWRITE_LLVM_ROOT_ENABLE to ON if you would like to overwrite.")
    endif()
    include(FetchContent)
    message(STATUS "LLVM: Building LLVM from source")

    set(LLVM_GIT_REPOSITORY https://github.com/llvm/llvm-project.git)
    set(LLVM_GIT_TAG llvmorg-9.0.1)

    FetchContent_Declare(
        llvm
        GIT_REPOSITORY ${LLVM_GIT_REPOSITORY}
        GIT_TAG ${LLVM_GIT_TAG}
        GIT_SHALLOW 1
        )

    FetchContent_GetProperties(llvm)
    if(NOT llvm_POPULATED)
        FetchContent_Populate(llvm)
    endif()

    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
        -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}
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
        -DLLVM_TARGETS_TO_BUILD=X86
        -DLLVM_ENABLE_BINDINGS=OFF
        -DLLVM_ENABLE_TERMINFO=OFF
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON
        -DLLVM_ENABLE_WARNINGS=OFF
        -DLLVM_ENABLE_PEDANTIC=OFF
        -DLIBOMP_OMPT_SUPPORT=OFF
        -DCLANG_ENABLE_ARCMT=OFF
        -DCLANG_ENABLE_STATIC_ANALYZER=OFF
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
    execute_process(COMMAND "${CMAKE_COMMAND}" --install .
        WORKING_DIRECTORY "${llvm_BINARY_DIR}")

    set(Clang_ROOT ${LLVM_ROOT})
    find_package(Clang REQUIRED CONFIG)
endif()

message(STATUS "CLANG_CMAKE_DIR: ${CLANG_CMAKE_DIR}")
message(STATUS "CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

add_library(libllvm INTERFACE)
target_include_directories(libllvm INTERFACE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIR})
target_link_libraries(libllvm INTERFACE clangHandleCXX clangHandleLLVM)
