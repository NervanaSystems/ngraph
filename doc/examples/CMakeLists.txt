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
        abc.cpp
        ${PROJECT_SOURCE_DIR}/doc/examples/abc.cpp
    )

    add_executable(abc ${SRC})
    add_dependencies(abc ngraph)

    set(HEADER_SEARCH_DEFINES
        "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    )

    target_link_libraries(abc ngraph)
    include_directories(SYSTEM ${JSON_INCLUDE_DIR})

    set_source_files_properties(abc.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

endif()
