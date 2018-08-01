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
    intelgpu_tensor_view.cpp
    intelgpu_layout.cpp
    intelgpu_op_batchnorm.cpp
    intelgpu_op_broadcast.cpp
    intelgpu_op_custom_kernels.cpp
    code_writer.cpp
)

if (NGRAPH_INTELGPU_ENABLE)
    add_library(intelgpu_backend SHARED ${SRC})
    set_target_properties(intelgpu_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_link_libraries(intelgpu_backend PUBLIC ngraph)
    find_package(CLDNN REQUIRED)
    target_include_directories(intelgpu_backend SYSTEM PUBLIC ${CLDNN_INCLUDE_DIRS})
    target_link_libraries(intelgpu_backend PUBLIC ${CLDNN_LIBRARIES})

    install(TARGETS intelgpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
