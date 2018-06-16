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

set(SRC
    intelgpu_backend.cpp
)

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    add_definitions(-DNGRAPH_DISTRIBUTED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    link_libraries(${MPI_CXX_LIBRARIES})
endif()

if (NGRAPH_INTELGPU_ENABLE)
    add_library(intelgpu_backend SHARED ${SRC})
    set_target_properties(intelgpu_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_link_libraries(intelgpu_backend PUBLIC ngraph)

    install(TARGETS intelgpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
