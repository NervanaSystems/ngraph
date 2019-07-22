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

if (NGRAPH_INTERPRETER_STATIC_LIB_ENABLE)
    set(LIBRARY_TYPE STATIC)
else()
    set(LIBRARY_TYPE SHARED)
endif()

if (NGRAPH_INTERPRETER_ENABLE)
    add_library(interpreter_backend ${LIBRARY_TYPE} int_backend.cpp node_wrapper.cpp int_executable.cpp)
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(interpreter_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(interpreter_backend PUBLIC ngraph)
    if(NGRAPH_INTERPRETER_STATIC_LIB_ENABLE)
        target_compile_definitions(interpreter_backend PUBLIC NGRAPH_INTERPRETER_STATIC_LIB_ENABLE)
    endif()
    install(TARGETS interpreter_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
