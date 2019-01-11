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

set (SRC
    main.cpp
    util.cpp
    uncomment.cpp
    header_rewrite.cpp
)

set(LLVM_LIB_DIR ${LLVM_INCLUDE_DIR}/../lib/clang/5.0.2/include)
get_filename_component(LLVM_LIB_DIR ${LLVM_LIB_DIR} ABSOLUTE)

list(APPEND HEADER_SEARCH_DEFINES CLANG_BUILTIN_HEADERS_PATH="${LLVM_LIB_DIR}")
list(APPEND HEADER_SEARCH_DEFINES NGRAPH_HEADERS_PATH="${NGRAPH_INCLUDE_PATH}")

if(NGRAPH_CPU_ENABLE)
    get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND HEADER_SEARCH_DEFINES EIGEN_HEADERS_PATH="${EIGEN_INCLUDE_DIR}")
    list(APPEND HEADER_SEARCH_DEFINES MKLDNN_HEADERS_PATH="${MKLDNN_INCLUDE_DIR}")
endif()

if(NGRAPH_TBB_ENABLE)
    list(APPEND HEADER_SEARCH_DEFINES TBB_HEADERS_PATH="${TBB_ROOT}/include")
endif()

add_executable(resource_generator EXCLUDE_FROM_ALL ${SRC})

if (NOT NGRAPH_DEX_ONLY)
    add_dependencies(resource_generator ext_llvm)
endif()

if(NGRAPH_CPU_ENABLE)
    add_dependencies(resource_generator ext_eigen ext_mkldnn)
endif()

set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

