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

set(HEADER_SEARCH_DEFINES
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\""
    "MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\""
    "CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.1/include\""
    "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    "INSTALLED_HEADERS_PATH=\"${CMAKE_INSTALL_PREFIX}/include\""
)
    set_source_files_properties(compiler.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES};")

if(NGRAPH_DISTRIBUTED_ENABLE AND MPI_CXX_INCLUDE_PATH)
    set_property(SOURCE compiler.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "MPI_HEADER_PATH=\"${MPI_C_INCLUDE_PATH}\";")
endif()

set_property(SOURCE compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
"CUDA_HEADER_PATHS=\"${CUDA_INCLUDE_DIRS}\";")
set_property(SOURCE compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
"CUDNN_HEADER_PATHS=\"${CUDNN_INCLUDE_DIRS}\";")

# Generate the resource file containing all headers used by the codegen compiler
add_custom_target(header_resource
    resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
    DEPENDS resource_generator ext_eigen ext_llvm ext_mkldnn
    BYPRODUCTS
)

add_library(codegen SHARED ${SRC})
add_dependencies(codegen header_resource ext_llvm)
target_include_directories(codegen PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(codegen PUBLIC ${LLVM_LINK_LIBS})
