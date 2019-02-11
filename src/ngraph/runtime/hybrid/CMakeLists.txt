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

add_library(hybrid_base STATIC
    hybrid_backend.cpp
    hybrid_util.cpp
    pass/default_placement.cpp
    pass/dump.cpp
    pass/fix_get_output_element.cpp
    pass/liveness.cpp
    pass/memory_layout.cpp
    )
target_link_libraries(hybrid_base PUBLIC ngraph)
set_target_properties(hybrid_base PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

install(TARGETS hybrid_base
    ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
)
