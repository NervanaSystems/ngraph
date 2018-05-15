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
set(ONNX_PROTO ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto)
set(ONNX_PB_H ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)
set(ONNX_PB_CC ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)

set(ONNX_IMPORT_SOURCE_FILES
        ${ONNX_PB_CC}
        graph.cpp
        mapping.cpp
        model.cpp
        node.cpp
        value_info.cpp
        onnx_util.cpp)

set(ONNX_IMPORT_HEADER_FILES
        ${ONNX_PB_H}
        graph.hpp
        mapping.hpp
        model.hpp
        node.hpp
        value_info.hpp
        onnx_util.hpp)

set_source_files_properties(${ONNX_PB_H} ${ONNX_PB_CC} PROPERTIES GENERATED TRUE)
file(DOWNLOAD ${ONNX_SOURCE_URL} ${ONNX_PROTO})

exec_program(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${ONNX_PROTO})

add_library(onnx_import STATIC ${ONNX_IMPORT_SOURCE_FILES} ${ONNX_IMPORT_HEADER_FILES} ${ONNX_PROTO})
target_include_directories(onnx_import PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${NGRAPH_INCLUDE_PATH})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
        -Wno-extended-offsetof -Wno-zero-as-null-pointer-constant -Wno-shorten-64-to-32 -Wno-unused-macros
        -Wno-missing-variable-declarations -Wno-unused-private-field)
endif()
