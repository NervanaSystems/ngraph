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

set (SRC
    nbench.cpp
    benchmark.cpp
)

add_executable(nbench ${SRC})

target_link_libraries(nbench ngraph)
if (NGRAPH_CPU_ENABLE)
    if (APPLE)
        target_link_libraries(nbench -Wl,-force_load cpu_backend)
    else()
        target_link_libraries(nbench -Wl,--whole-archive cpu_backend -Wl,--no-whole-archive)
    endif()
endif()
if (NGRAPH_INTELGPU_ENABLE)
    target_link_libraries(nbench intelgpu_backend)
endif()
if (NGRAPH_GPU_ENABLE)
    target_link_libraries(nbench gpu_backend)
endif()
if (NGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(nbench interpreter_backend)
endif()

install(TARGETS nbench RUNTIME DESTINATION ${NGRAPH_INSTALL_BIN})
