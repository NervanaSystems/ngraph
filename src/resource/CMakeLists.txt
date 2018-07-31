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

set (SRC
    main.cpp
    util.cpp
    uncomment.cpp
    header_rewrite.cpp
)

get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LLVM_INCLUDE_DIR libllvm INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(TVM_INCLUDE_DIR libtvm INTERFACE_INCLUDE_DIRECTORIES)
set(CLANG_INCLUDE_DIR ${LLVM_INCLUDE_DIR}/../lib/clang/5.0.1/include)
get_filename_component(CLANG_INCLUDE_DIR ${CLANG_INCLUDE_DIR} ABSOLUTE)

set(HEADER_SEARCH_DEFINES
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\""
    "MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\""
    "CLANG_BUILTIN_HEADERS_PATH=\"${CLANG_INCLUDE_DIR}\""
    "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    "TVM_HEADERS_PATH=\"${TVM_INCLUDE_DIR}\""
)

if(NGRAPH_TBB_ENABLE)
    list(APPEND HEADER_SEARCH_DEFINES "TBB_HEADERS_PATH=\"${TBB_ROOT}/include\"")
    list(APPEND HEADER_SEARCH_DEFINES "NGRAPH_TBB_ENABLE")
endif()

if (NGRAPH_GPU_ENABLE)
    find_package(CUDA 8 REQUIRED)
    find_package(CUDNN 7 REQUIRED)
    set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
    "CUDA_HEADER_PATHS=\"${CUDA_INCLUDE_DIRS}\";")
    set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
    "CUDNN_HEADER_PATHS=\"${CUDNN_INCLUDE_DIRS}\";")
endif()

add_executable(resource_generator EXCLUDE_FROM_ALL ${SRC})
add_dependencies(resource_generator ext_llvm ext_eigen ext_mkldnn)

set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

