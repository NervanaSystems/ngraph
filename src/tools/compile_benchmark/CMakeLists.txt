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

if(MKLDNN_INCLUDE_DIR)
    link_directories(${MKLDNN_LIB_DIR})
endif()

if (NGRAPH_CPU_ENABLE)
    set (SRC
        compile_benchmark.cpp
        ${PROJECT_SOURCE_DIR}/test/util/benchmark.cpp
    )

    add_executable(compile_benchmark ${SRC})
    add_dependencies(compile_benchmark ngraph)

    set(HEADER_SEARCH_DEFINES
        "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    )

    target_link_libraries(compile_benchmark ngraph)
    include_directories("${PROJECT_SOURCE_DIR}/test")

    set_source_files_properties(compile_benchmark.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
endif()
