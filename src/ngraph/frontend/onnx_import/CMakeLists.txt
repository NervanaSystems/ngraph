# ******************************************************************************
# Copyright (c) 2017-2018 Intel Copyright
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by application law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(SOURCE_URL "https://raw.githubusercontent.com/onnx/onnx/v1.1.2/onnx")
set(ONNX_URL "${SOURCE_URL}/onnx.proto")
set(OPERATORS_URL "${SOURCE_URL}/onnx-operators.proto")

set_source_files_properties(onnx.pb.h onnx.pb.cc onnx-operators.pb.h onnx-operators.pb.cc PROPERTIES GENERATED TRUE)

if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnx.proto")
    file(DOWNLOAD ${ONNX_URL} ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto)
endif()

if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnx-operators.poto")
    file(DOWNLOAD ${OPERATORS_URL} ${CMAKE_CURRENT_BINARY_DIR}/onnx-operators.proto)
endif()

add_custom_command(OUTPUT onnx.pb.cc onnx.pb.h
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto
    DEPENDS onnx.proto)

add_custom_command(OUTPUT onnx-operators.pb.cc onnx-operators.pb.h
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx-operators.proto
    DEPENDS onnx-operators.proto)

add_library(onnx_import_interface OBJECT
    onnx.cpp
    onnx.hpp
    onnx.pb.h
    onnx-operators.pb.h
    onnx.proto
    onnx-operators.proto)

add_library(onnx_import STATIC
    onnx.pb.cc
    onnx-operators.pb.cc
    attribute.cpp
    attribute.hpp
    graph.cpp
    graph.hpp
    model.hpp
    node.cpp
    node.hpp
    op/add.hpp
    op/constant.hpp
    op/conv.hpp
    op/split.hpp
    ops_bridge.cpp
    ops_bridge.hpp
    tensor.hpp
    value_info.hpp)

add_dependencies(onnx_import onnx_import_interface)

set_property(TARGET onnx_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import PUBLIC ${CMAKE_CURRENT_BINARY_DIR} PRIVATE ${NGRAPH_INCLUDE_PATH})
target_link_libraries(onnx_import PRIVATE ${PROTOBUF_LIBRARIES})

set_property(TARGET onnx_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import_interface PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${NGRAPH_INCLUDE_PATH})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
        -Wno-extended-offsetof -Wno-zero-as-null-pointer-constant -Wno-shorten-64-to-32 -Wno-unused-macros
        -Wno-missing-variable-declarations -Wno-unused-private-field)
    target_compile_options(onnx_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
        -Wno-extended-offsetof -Wno-zero-as-null-pointer-constant -Wno-shorten-64-to-32 -Wno-unused-macros
        -Wno-missing-variable-declarations -Wno-unused-private-field)
endif()

