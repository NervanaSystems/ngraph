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

if(NOT NGRAPH_CODEGEN_ENABLE)
    return()
endif()

if (DEFINED NGRAPH_USE_CXX_ABI AND NGRAPH_USE_PREBUILT_LLVM)
    message(FATAL_ERROR "Unable to use NGRAPH_USE_PREBUILT_LLVM with NGRAPH_USE_CXX_ABI")
endif()

set(SRC
    compiler.cpp
    execution_engine.cpp
)
add_library(codegen SHARED ${SRC})

# LLVM binary builds are typically built without RTTI
# The built-in headers are in a version-specific directory
# This must be kept in sync with the LLVM + Clang version in use
if(NOT WIN32)
   set_source_files_properties(compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

get_target_property(LLVM_INCLUDE_DIR libllvm INTERFACE_INCLUDE_DIRECTORIES)

# find_file(HEADER_1 cmath HINTS /usr/include/c++/7)
get_filename_component(CLANG_INCLUDE_DIR ${LLVM_INCLUDE_DIR}/../lib/clang/${NGRAPH_LLVM_VERSION}/include ABSOLUTE)

if(NGRAPH_CPU_ENABLE)
    get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND HEADER_SEARCH_DEFINES EIGEN_HEADERS_PATH="${EIGEN_INCLUDE_DIR}")
    list(APPEND HEADER_SEARCH_DEFINES MKLDNN_HEADERS_PATH="${MKLDNN_INCLUDE_DIR}")
endif()

list(APPEND HEADER_SEARCH_DEFINES CLANG_BUILTIN_HEADERS_PATH="${CLANG_INCLUDE_DIR}")
list(APPEND HEADER_SEARCH_DEFINES NGRAPH_HEADERS_PATH="${NGRAPH_INCLUDE_PATH}")

if(NGRAPH_TBB_ENABLE)
    get_target_property(TBB_INCLUDE_DIR libtbb INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND HEADER_SEARCH_DEFINES TBB_HEADERS_PATH="${TBB_INCLUDE_DIR}")
endif()

set_source_files_properties(compiler.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

# Generate the resource file containing all headers used by the codegen compiler
add_custom_target(header_resource
    resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
    DEPENDS resource_generator
    BYPRODUCTS
)
if (NGRAPH_CPU_ENABLE)
    add_dependencies(header_resource ext_eigen libmkldnn)
endif()

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(codegen PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()
add_dependencies(codegen header_resource)
if (NGRAPH_CPU_ENABLE)
    add_dependencies(codegen libmkldnn libeigen)
    if(LINUX)
        # --exclude-libs=ALL prevents symbols from statically-linked libraries (LLVM, in this case)
        # from being automatically exported
        set_property(TARGET codegen APPEND PROPERTY LINK_FLAGS "-Wl,--exclude-libs=ALL")
    endif()
endif()
target_include_directories(codegen SYSTEM PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(codegen PRIVATE libllvm ngraph)
install(TARGETS codegen DESTINATION ${NGRAPH_INSTALL_LIB})
