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
    intelgpu_backend.cpp
    intelgpu_tensor_view.cpp
    intelgpu_layout.cpp
    intelgpu_op_batchnorm.cpp
    intelgpu_op_broadcast.cpp
    intelgpu_op_custom_kernels.cpp
    intelgpu_op_convolution.cpp
    intelgpu_op_softmax.cpp
    intelgpu_op_custom_func_call.cpp
    code_writer.cpp
    visualize_tree.cpp
)

if (NGRAPH_INTELGPU_ENABLE)
    add_library(intelgpu_backend SHARED ${SRC})
    target_link_libraries(intelgpu_backend PUBLIC ngraph libcldnn)

    if (NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(intelgpu_backend
            PROPERTIES
                VERSION ${NGRAPH_VERSION}
                SOVERSION ${NGRAPH_API_VERSION})
    endif()

    set_target_properties(intelgpu_backend
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS intelgpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
