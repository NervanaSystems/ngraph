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

add_library(onnx_import_interface OBJECT
        onnx.cpp
        onnx.hpp)

target_include_directories(onnx_import_interface PRIVATE ${ONNX_PROTO_INCLUDE_DIR})

add_library(onnx_import STATIC
        core/attribute.cpp
        core/attribute.hpp
        core/graph.cpp
        core/graph.hpp
        core/model.hpp
        core/node.cpp
        core/node.hpp
        core/tensor.hpp
        core/value_info.hpp
        exceptions.hpp
        op/add.hpp
        op/average_pool.cpp
        op/average_pool.hpp
        op/batch_norm.cpp
        op/batch_norm.hpp
        op/constant.cpp
        op/constant.hpp
        op/conv.cpp
        op/gemm.cpp
        op/gemm.hpp
        op/matmul.hpp
        op/max_pool.cpp
        op/max_pool.hpp
        op/mul.hpp
        op/relu.hpp
        op/split.cpp
        op/split.hpp
        ops_bridge.cpp
        utils/broadcasting.cpp
        utils/broadcasting.hpp
        utils/convpool.cpp
        utils/convpool.hpp
        utils/reshape.cpp
        utils/reshape.hpp)

add_dependencies(onnx_import onnx_import_interface)

if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
    add_dependencies(onnx_import_interface protobuf::libprotobuf)
    add_dependencies(onnx_import_interface onnx::libonnx_proto)
endif()

set_property(TARGET onnx_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import PUBLIC ${CMAKE_CURRENT_BINARY_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${NGRAPH_INCLUDE_PATH} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIR})
target_link_libraries(onnx_import PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY})

set_property(TARGET onnx_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import_interface PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    ${NGRAPH_INCLUDE_PATH} ${Protobuf_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
    target_compile_options(onnx_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
endif()
