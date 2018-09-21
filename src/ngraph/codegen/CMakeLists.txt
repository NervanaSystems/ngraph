# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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

set(SRC
    code_writer.cpp
    compiler.cpp
    execution_engine.cpp
)

# LLVM binary builds are typically built without RTTI
# The built-in headers are in a version-specific directory
# This must be kept in sync with the LLVM + Clang version in use
set_source_files_properties(compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LLVM_LIB_DIR libllvm INTERFACE_INCLUDE_DIRECTORIES)

# find_file(HEADER_1 cmath HINTS /usr/include/c++/7)
get_filename_component(LLVM_LIB_DIR ${LLVM_LIB_DIR}/../lib/clang/5.0.2/include ABSOLUTE)

set(HEADER_SEARCH_DEFINES
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\""
    "MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\""
    "CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}\""
    "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
)

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

    string(REPLACE ";" "\\;" MPI_C_INCLUDE_PATH "${MPI_C_INCLUDE_PATH}")
    add_definitions("-DMPI_HEADER_PATH=\"${MPI_C_INCLUDE_PATH}\"")
    add_definitions(-DNGRAPH_DISTRIBUTED)
endif()

if(NGRAPH_GPU_ENABLE)
    find_package(CUDA 8 REQUIRED)
    find_package(CUDNN 7 REQUIRED)
    list(APPEND HEADER_SEARCH_DEFINES "CUDA_HEADER_PATHS=\"${CUDA_INCLUDE_DIRS}\"")
    list(APPEND HEADER_SEARCH_DEFINES "CUDNN_HEADER_PATHS=\"${CUDNN_INCLUDE_DIRS}\"")
endif()

if(NGRAPH_TBB_ENABLE)
    list(APPEND HEADER_SEARCH_DEFINES "TBB_HEADERS_PATH=\"${TBB_ROOT}/include\"")
endif()

set_source_files_properties(compiler.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

# Generate the resource file containing all headers used by the codegen compiler
add_custom_target(header_resource
    resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
    DEPENDS resource_generator ext_eigen ext_llvm ext_mkldnn
    BYPRODUCTS
)

if (NGRAPH_GPU_ENABLE OR (NGRAPH_CPU_ENABLE AND NOT NGRAPH_DEX_ONLY))
    add_library(codegen SHARED ${SRC})
    set_target_properties(codegen PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    add_dependencies(codegen header_resource libmkldnn libeigen)
    target_include_directories(codegen SYSTEM PUBLIC ${CMAKE_BINARY_DIR})
    target_link_libraries(codegen PRIVATE libllvm ngraph)
    install(TARGETS codegen DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
