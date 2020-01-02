# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

add_library(onnxifi-ngraph SHARED
    onnxifi.cpp
    backend.hpp
    backend_manager.hpp
    backend_manager.cpp
    exceptions.hpp
    span.hpp
    tensor.hpp
    tensor.cpp)

target_link_libraries(onnxifi-ngraph PRIVATE ngraph)

add_dependencies(onnxifi-ngraph onnx::libonnx)
target_include_directories(onnxifi-ngraph SYSTEM PRIVATE ${ONNX_INCLUDE_DIR} ${ONNX_IMPORT_INCLUDE_DIR})

set(ONNXIFI_VERSION ${NGRAPH_VERSION})
set(ONNXIFI_ABI_VERSION 1)

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(onnxifi-ngraph PROPERTIES
        VERSION ${ONNXIFI_VERSION}
        SOVERSION ${ONNXIFI_ABI_VERSION})
endif()

if(WIN32)
    if(MSVC)
        target_compile_definitions(onnxifi-ngraph PRIVATE "ONNXIFI_PUBLIC=__declspec(dllexport)")
    else()
        target_compile_definitions(onnxifi-ngraph PRIVATE "ONNXIFI_PUBLIC=__attribute__((__dllexport__))")
    endif()
endif()

install(TARGETS onnxifi-ngraph DESTINATION ${NGRAPH_INSTALL_LIB})
