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

cmake_minimum_required (VERSION 3.1)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed.")
endif()

if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    message(FATAL_ERROR "Ninja generator not supported.")
endif()

if (CMAKE_BUILD_TYPE)
    set(RELEASE_TYPES Debug Release RelWithDebInfo MinSizeRel)
    list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
    if (${INDEX_FOUND} EQUAL -1)
        message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel")
    endif()
endif()

include(git_tags)
include(var_functions)

NGRAPH_GET_VERSION_LABEL()

string(REGEX MATCH "([0-9?]+)\.([0-9?]+)\.([0-9?]+)" NGRAPH_VERSION_SHORT "${NGRAPH_VERSION_LABEL}")
string(REGEX MATCH "([0-9?]+)\.([0-9?]+)" NGRAPH_API_VERSION "${NGRAPH_VERSION_LABEL}")
string(REGEX MATCH "[^v](.*)" NGRAPH_VERSION "${NGRAPH_VERSION_LABEL}")
string(REPLACE "." ";" NGRAPH_VERSION_PARTS "${NGRAPH_VERSION_SHORT}")
list(GET NGRAPH_VERSION_PARTS 0 NGRAPH_VERSION_MAJOR)
list(GET NGRAPH_VERSION_PARTS 1 NGRAPH_VERSION_MINOR)
list(GET NGRAPH_VERSION_PARTS 2 NGRAPH_VERSION_PATCH)
configure_file(VERSION.in VERSION)

message(STATUS "NGRAPH_VERSION ${NGRAPH_VERSION}")
message(STATUS "NGRAPH_VERSION_SHORT ${NGRAPH_VERSION_SHORT}")
message(STATUS "NGRAPH_API_VERSION ${NGRAPH_API_VERSION}")

set(NGRAPH_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (APPLE)
    # Suppress an OS X-specific warning.
    if (POLICY CMP0042)
        cmake_policy(SET CMP0042 NEW)
    else()
        set(MACOS_RPATH ON)
    endif()
endif()

project (ngraph)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if ("${CMAKE_GENERATOR}" MATCHES "^Visual Studio.*$")
    set(MSVS TRUE)
endif()

# Setup CMAKE_ARGS to be forwarded to External Projects
set(NGRAPH_FORWARD_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        )

if (NOT MSVS)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    endif()

    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )
endif()
message(STATUS "NGRAPH_FORWARD_CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}")

# Prevent Eigen from using any LGPL3 code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY -DTBB_USE_THREADING_TOOLS")

ngraph_var(NGRAPH_USE_PREBUILT_LLVM DEFAULT "FALSE")

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" TRUE)
option(NGRAPH_TOOLS_ENABLE "Control the building of tool" TRUE)
option(NGRAPH_CPU_ENABLE "Control the building of the CPU backend" TRUE)
option(NGRAPH_INTELGPU_ENABLE "Control the building of the Intel GPU backend with clDNN" FALSE)
option(NGRAPH_GPU_ENABLE "Control the building of the GPU backend" FALSE)
option(NGRAPH_INTERPRETER_ENABLE "Control the building of the INTERPRETER backend" TRUE)
option(NGRAPH_NOP_ENABLE "Control the building of the NOP backend" TRUE)
option(NGRAPH_GPUH_ENABLE "Control the building of the Hybrid GPU backend" FALSE)
option(NGRAPH_GENERIC_CPU_ENABLE "Enable build nGraph for generic CPU backend" FALSE)
option(NGRAPH_DISTRIBUTED_ENABLE "Add distributed mode to the CPU backend" FALSE)
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" FALSE)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" FALSE)
option(NGRAPH_DEX_ONLY "Build CPU DEX without codegen" FALSE)
option(NGRAPH_CODE_COVERAGE_ENABLE "Enable code coverage data collection" FALSE)
option(NGRAPH_LIB_VERSIONING_ENABLE "Enable shared library versioning" FALSE)
option(NGRAPH_PYTHON_BUILD_ENABLE "Enable build nGraph python package wheel" FALSE)
option(NGRAPH_PLAIDML_ENABLE "Enable the PlaidML backend" ${PLAIDML_FOUND})

if (NGRAPH_GPUH_ENABLE)
    set(NGRAPH_GPU_ENABLE TRUE)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" FALSE)
    option(NGRAPH_ONNXIFI_ENABLE "Enable ONNX Interface for Framework Integration" TRUE)
endif()

message(STATUS "NGRAPH_UNIT_TEST_ENABLE:      ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_TOOLS_ENABLE:          ${NGRAPH_TOOLS_ENABLE}")
message(STATUS "NGRAPH_CPU_ENABLE:            ${NGRAPH_CPU_ENABLE}")
message(STATUS "NGRAPH_INTELGPU_ENABLE:       ${NGRAPH_INTELGPU_ENABLE}")
message(STATUS "NGRAPH_GPU_ENABLE:            ${NGRAPH_GPU_ENABLE}")
message(STATUS "NGRAPH_INTERPRETER_ENABLE:    ${NGRAPH_INTERPRETER_ENABLE}")
message(STATUS "NGRAPH_NOP_ENABLE:            ${NGRAPH_NOP_ENABLE}")
message(STATUS "NGRAPH_GPUH_ENABLE:           ${NGRAPH_GPUH_ENABLE}")
message(STATUS "NGRAPH_GENERIC_CPU_ENABLE:    ${NGRAPH_GENERIC_CPU_ENABLE}")
message(STATUS "NGRAPH_DISTRIBUTED_ENABLE:    ${NGRAPH_DISTRIBUTED_ENABLE}")
message(STATUS "NGRAPH_DEBUG_ENABLE:          ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_ONNX_IMPORT_ENABLE:    ${NGRAPH_ONNX_IMPORT_ENABLE}")
message(STATUS "NGRAPH_DEX_ONLY:              ${NGRAPH_DEX_ONLY}")
message(STATUS "NGRAPH_CODE_COVERAGE_ENABLE:  ${NGRAPH_CODE_COVERAGE_ENABLE}")
message(STATUS "NGRAPH_LIB_VERSIONING_ENABLE: ${NGRAPH_LIB_VERSIONING_ENABLE}")
message(STATUS "NGRAPH_PYTHON_BUILD_ENABLE:   ${NGRAPH_PYTHON_BUILD_ENABLE}")
message(STATUS "NGRAPH_USE_PREBUILT_LLVM:     ${NGRAPH_USE_PREBUILT_LLVM}")
message(STATUS "NGRAPH_PLAIDML_ENABLE:        ${NGRAPH_PLAIDML_ENABLE}")

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_BINDIR "bin")
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_DOCDIR "doc")
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# Destinations
set(NGRAPH_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
if (LINUX)
    if (DEFINED NGRAPH_RPATH)
        set(CMAKE_INSTALL_RPATH "$ORIGIN:${NGRAPH_RPATH}")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()
endif()

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------

# Compiler-specific logic...
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    message( STATUS "Setting clang flags...")
    include( cmake/clang_4_0_flags.cmake )
endif()

if (WIN32)
    set (CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_FLAGS "/W0 /EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    # These can be uncommented once we have visibility fully in place
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

include(cmake/sdl.cmake)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        message( STATUS "nGraph using CXX11 ABI: " ${NGRAPH_USE_CXX_ABI} )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    endif()
endif()

ngraph_var(NGRAPH_WARNINGS_AS_ERRORS DEFAULT "OFF")
if (${NGRAPH_WARNINGS_AS_ERRORS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    message(STATUS "Warnings as errors")
endif()

if (NGRAPH_CODE_COVERAGE_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

# Enable build target CPU features
if(NOT WIN32)
    set(NGRAPH_TARGET_ARCH native CACHE
        STRING "Target CPU architecture to build for. Defaults to the native CPU architecture")

    if (NOT "${NGRAPH_TARGET_ARCH}" STREQUAL "native")
        message(WARNING
            "Build target architecture was overridden. The resulting build might not work correctly on the host CPU.")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${NGRAPH_TARGET_ARCH}")
endif()

if (DEFINED NGRAPH_TUNE_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=${NGRAPH_TUNE_ARCH}")
endif()

if (NGRAPH_USE_GOLD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

if (NGRAPH_CPU_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_CPU_ENABLE")
endif()

if (NGRAPH_PLAIDML_ENABLE)
    find_package(PlaidML CONFIG)
    if (NOT PLAIDML_FOUND)
        message(FATAL_ERROR "PlaidML not installed")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_PlaidML_ENABLE")
endif()

if (NOT DEFINED NGRAPH_TBB_ENABLE)
    set(NGRAPH_TBB_ENABLE ${NGRAPH_CPU_ENABLE})
endif()

add_custom_target(style-check
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    -P ${CMAKE_MODULE_PATH}style_check.cmake
)

add_custom_target(style-apply
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    -P ${CMAKE_MODULE_PATH}style_apply.cmake
)

add_custom_target(fix-mode
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    -P ${CMAKE_MODULE_PATH}fix_mode.cmake
)

#-----------------------------------------------------------------------------------------------
# enable or disable output from NGRAPH_DEBUG statements
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------

set(NGRAPH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/ngraph)
if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
endif()

set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        include(cmake/external_protobuf.cmake)
    else()
        find_package(Protobuf 2.6.1 REQUIRED)
    endif()
    include(cmake/external_onnx.cmake)
    if (TARGET ext_protobuf)
        add_dependencies(ext_onnx ext_protobuf)
    endif()
endif()

include(cmake/external_gtest.cmake)
include(cmake/external_json.cmake)
include(cmake/external_eigen.cmake)
include(cmake/external_mkldnn.cmake)
include(cmake/external_cldnn.cmake)

if (NGRAPH_USE_PREBUILT_LLVM OR DEFINED LLVM_TARBALL_URL)
    include(cmake/external_llvm_prebuilt.cmake)
elseif (NOT NGRAPH_DEX_ONLY OR NOT MSVS)
    include(cmake/external_llvm.cmake)
endif()

if (WIN32 OR APPLE)
    include(cmake/external_tbb_prebuilt.cmake)
else()
    include(cmake/external_tbb.cmake)
endif()
if (NGRAPH_DISTRIBUTED_ENABLE)
    include(cmake/external_mlsl.cmake)
endif()


if (NGRAPH_HALIDE)
    message(WARNING "Halide build system integration is currently using an older LLVM release \
                     and is not expected to work across most build environments. Consider \
                     disabling it till this message goes away")
    include(cmake/external_halide.cmake)
endif()

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
add_subdirectory(src)

if (NGRAPH_UNIT_TEST_ENABLE)
    add_subdirectory(test)
    message(STATUS "unit tests enabled")
else()
    add_subdirectory(test/models)
    add_subdirectory(test/util)
    message(STATUS "unit tests disabled")
endif()

if (NGRAPH_DOC_BUILD_ENABLE)
    add_subdirectory(doc)
endif()

if (NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(python)
endif()

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/licenses
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_PREFIX})
