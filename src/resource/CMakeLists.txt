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

if(NOT NGRAPH_CODEGEN_ENABLE)
    return()
endif()

if(LINUX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

set (SRC
    main.cpp
    util.cpp
    uncomment.cpp
    header_rewrite.cpp
)

add_executable(resource_generator EXCLUDE_FROM_ALL ${SRC})
add_dependencies(resource_generator libllvm)
get_filename_component(CLANG_INTRIN_INCLUDE_DIR ${LLVM_INCLUDE_DIRS}/../lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include ABSOLUTE)
list(APPEND HEADER_SEARCH_DEFINES CLANG_BUILTIN_HEADERS_PATH="${CLANG_INTRIN_INCLUDE_DIR}")
list(APPEND HEADER_SEARCH_DEFINES NGRAPH_HEADERS_PATH="${NGRAPH_INCLUDE_PATH}")

set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

