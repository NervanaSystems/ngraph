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

set(LLVM_REPO_URL https://github.com/llvm/llvm-project.git)

# Change these commit IDs to move to latest stable versions
#set(LLVM_COMMIT_ID 3c5dd5863c34ecd51e9d2a49929877d8151dea39)
set(LLVM_COMMIT_ID 663860f63e73518fc09e55a4a68b03f8027eafc8)

# If prebuilt path is given check if it is already populated with
# correct with a build with commit id and cmake config files
# otherwise build commit id and install to prebuilt path
set(BUILD_LLVM TRUE)

if(${LLVM_PREBUILT_PATH})
    set(VCSREVISION "${LLVM_PREBUILT_PATH}/include/llvm/Support/VCSRevision.h")
    if(EXISTS "${VCSREVISION}")
        file(READ "${VCSREVISION}" REVISION_FILE)
        string(REGEX MATCH "LLVM_REVISION \"([A-Za-z0-9]+)\"" _ ${REVISION_FILE})
        set(LONG_REV ${CMAKE_MATCH_1})
        string(TOLOWER ${LONG_REV} LONG_REV)
        string(TOLOWER ${COMMIT_ID} COMMIT_ID)
        if(LONG_REV STREQUAL COMMIT_ID)
            message(STATUS "SHA1 HASH Matches.")
            set(BUILD_LLVM FALSE)
        endif()
    endif()
endif()

# MLIR environment variables. Some of them are used by LIT tool.

# Only used in this file
set(MLIR_LLVM_ROOT ${MLIR_PROJECT_ROOT}/llvm-project)
set(MLIR_LLVM_SOURCE_DIR ${MLIR_LLVM_ROOT}/llvm)
set(MLIR_SOURCE_DIR ${MLIR_LLVM_ROOT}/mlir)
# Used in test/mlir:
# lit cfg
set(MLIR_LLVM_BUILD_DIR ${MLIR_PROJECT_ROOT}/build)
set(NGRAPH_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(NGRAPH_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/mlir)
# lit cfg and path to llvm-lit
set(MLIR_LLVM_TOOLS_DIR ${MLIR_LLVM_BUILD_DIR}/bin)

# MLIR has to be pre-built before ngraph build starts
# this will clone and build MLIR during cmake config instead

# we will fetch and build it from the source if cmake is not configured to use
# the prebuilt mlir
if (NOT NGRAPH_USE_PREBUILT_MLIR)
    FetchContent_Declare(
        ext_llvm
        GIT_REPOSITORY ${LLVM_REPO_URL}
        GIT_TAG        ${LLVM_COMMIT_ID}
    )
    FetchContent_GetProperties(ext_llvm)
    if(NOT ext_llvm_POPULATED)
        FetchContent_Populate(ext_llvm)
        #add_subdirectory(${ext_llvm_SOURCE_DIR} ${ext_llvm_BINARY_DIR})
    endif()

    # set llvm build options
    set(LLVM_CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}
                   -DLLVM_ENABLE_RTTI=ON
                   -DLLVM_ENABLE_PROJECTS:STRING=mlir
                   -DLLVM_BUILD_EXAMPLES=ON
                   -DLLVM_TARGETS_TO_BUILD=host)

    # configure llvm
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                    -DCMAKE_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}
                    -DCMAKE_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}
                    ${MLIR_LLVM_CMAKE_ARGS}
                    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_ORIGINAL_CXX_FLAGS}
                    ${ext_llvm_SOURCE_DIR}/llvm
                    WORKING_DIRECTORY "${ext_llvm_BINARY_DIR}")

    # build llvm.
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${ext_llvm_BINARY_DIR}")
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

set(MLIR_DIR "${MLIR_LLVM_BUILD_DIR}/lib/cmake/mlir"
    CACHE PATH "Path to MLIR cmake modules")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)

# Used by tblgen
set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
set(MLIR_BIN_INCLUDE_PATH ${MLIR_LLVM_BUILD_DIR}/tools/mlir/include)
# Used by ngraph mlir and cpu backend
set(MLIR_INCLUDE_PATHS ${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH})
set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR headers at: ${MLIR_INCLUDE_PATHS}")
message(STATUS "LLVM headers at: ${MLIR_LLVM_INCLUDE_PATH}")
