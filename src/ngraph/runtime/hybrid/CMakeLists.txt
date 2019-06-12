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

set(SRC
    hybrid_backend.cpp
    hybrid_backend.hpp
    hybrid_executable.cpp
    hybrid_executable.hpp
    hybrid_tensor.cpp
    hybrid_util.cpp
    hybrid_util.hpp
    op/function_call.cpp
    op/function_call.hpp
    pass/default_placement.cpp
    pass/default_placement.hpp
    pass/dump.cpp
    pass/dump.hpp
    pass/fix_get_output_element.cpp
    pass/fix_get_output_element.hpp
    pass/liveness.cpp
    pass/liveness.hpp
    pass/memory_layout.cpp
    pass/memory_layout.hpp
    )

add_library(hybrid STATIC ${SRC})
target_link_libraries(hybrid PUBLIC ngraph)

install(TARGETS hybrid
    ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB}
    LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB}
    )
