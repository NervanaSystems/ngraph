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

set(ONNX_SOURCE_URL "https://raw.githubusercontent.com/onnx/onnx/v1.1.2/onnx/onnx.proto")

set_source_files_properties(onnx.pb.h onnx.pb.cc PROPERTIES GENERATED TRUE)

add_custom_command(OUTPUT onnx.proto
        COMMAND wget --timeout=60 ${ONNX_SOURCE_URL})

add_custom_command(OUTPUT onnx.pb.cc onnx.pb.h
        COMMAND protobuf::protoc --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto
        DEPENDS onnx.proto)

add_library(onnx_import STATIC
        onnx.pb.cc
        onnx.pb.h
        onnx.proto
        graph.cpp
        mapping.cpp
        model.cpp
        node.cpp
        ops_bridge.cpp
        onnx_util.cpp
        tensor.cpp
        value_info.cpp
        graph.hpp
        mapping.hpp
        model.hpp
        node.hpp
        ops_bridge.hpp
        onnx_util.hpp
        tensor.hpp
        value_info.hpp)

target_include_directories(onnx_import
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE ${NGRAPH_INCLUDE_PATH})

target_link_libraries(onnx_import PRIVATE protobuf::libprotobuf)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
        -Wno-extended-offsetof -Wno-zero-as-null-pointer-constant -Wno-shorten-64-to-32 -Wno-unused-macros
        -Wno-missing-variable-declarations -Wno-unused-private-field)
endif()

