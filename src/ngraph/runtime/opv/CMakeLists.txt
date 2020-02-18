# ******************************************************************************
# Copyright 2020 Intel Corporation
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


if (NGRAPH_OPV_ENABLE)
    add_library(opv_backend ${LIBRARY_TYPE} opv_backend.cpp opv_executable.cpp)
    target_compile_definitions(opv_backend PRIVATE OPV_BACKEND_EXPORTS)
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(opv_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(opv_backend PUBLIC ngraph)
    install(TARGETS opv_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
