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

if (NGRAPH_GENERIC_CPU_ENABLE)
    # find_package(OpenMP)
    # if (OPENMP_FOUND)
    #     add_compile_options(${OpenMP_CXX_FLAGS})
    # endif()
    add_library(gcpu_backend SHARED gcpu_backend.cpp gcpu_executable.cpp node_wrapper.cpp)
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(gcpu_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(gcpu_backend PRIVATE ngraph libeigen)

    install(TARGETS gcpu_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
