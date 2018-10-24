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

add_library(onnxifi-ngraph SHARED
    onnxifi.cpp
    backend.hpp
    backend.cpp
    backend_manager.hpp
    backend_manager.cpp
    graph.hpp
    graph.cpp
    event_manager.hpp
    event_manager.cpp
    event.hpp
    span.hpp
    tensor.hpp
    tensor.cpp
    graph_manager.hpp
    graph_manager.cpp
    queue.hpp)

target_link_libraries(onnxifi-ngraph PRIVATE ngraph)

add_dependencies(onnxifi-ngraph onnx::libonnx)
target_include_directories(onnxifi-ngraph SYSTEM PRIVATE ${ONNX_INCLUDE_DIR} ${ONNX_IMPORT_INCLUDE_DIR})

set(ONNXIFI_VERSION ${NGRAPH_VERSION})
set(ONNXIFI_ABI_VERSION 1)

set_target_properties(onnxifi-ngraph PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libonnxifi-ngraph.version"
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libonnxifi-ngraph.version
    VERSION ${ONNXIFI_VERSION} SOVERSION ${ONNXIFI_ABI_VERSION})

target_compile_definitions(onnxifi-ngraph PRIVATE
    NGRAPH_VERSION="${NGRAPH_VERSION}" ONNX_VERSION="${ONNX_VERSION}"
    ONNX_OPSET_VERSION="${ONNX_OPSET_VERSION}")

install(TARGETS onnxifi-ngraph DESTINATION ${NGRAPH_INSTALL_LIB})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnxifi-ngraph PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated -Wno-unused-member-function)
endif()
