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

set(ONNX_OPSET_VERSION 11 CACHE INTERNAL "Supported version of ONNX operator set")

add_library(onnx_import_interface OBJECT
        core/node.cpp
        core/node.hpp
        core/operator_set.hpp
        core/weight.hpp
        onnx.cpp
        onnx.hpp)

add_library(onnx_import STATIC
        core/attribute.cpp
        core/attribute.hpp
        core/graph.cpp
        core/graph.hpp
        core/model.cpp
        core/model.hpp
        core/node.hpp
        core/null_node.cpp
        core/null_node.hpp
        core/operator_set.hpp
        core/tensor.hpp
        core/value_info.hpp
        exceptions.cpp
        exceptions.hpp
        op/acos.hpp
        op/acosh.cpp
        op/acosh.hpp
        op/add.hpp
        op/and.hpp
        op/argmax.cpp
        op/argmax.hpp
        op/argmin.cpp
        op/argmin.hpp
        op/asin.hpp
        op/asinh.cpp
        op/asinh.hpp
        op/atan.hpp
        op/atanh.cpp
        op/atanh.hpp
        op/average_pool.cpp
        op/average_pool.hpp
        op/batch_norm.cpp
        op/batch_norm.hpp
        op/cast.cpp
        op/cast.hpp
        op/ceil.hpp
        op/clip.cpp
        op/clip.hpp
        op/cos.hpp
        op/cosh.hpp
        op/concat.cpp
        op/concat.hpp
        op/constant.cpp
        op/constant.hpp
        op/conv.cpp
        op/conv.hpp
        op/conv_integer.cpp
        op/conv_integer.hpp
        op/conv_transpose.cpp
        op/conv_transpose.hpp
        op/depth_to_space.cpp
        op/depth_to_space.hpp
        op/dequantize_linear.cpp
        op/dequantize_linear.hpp
        op/div.hpp
        op/dropout.hpp
        op/elu.cpp
        op/elu.hpp
        op/equal.hpp
        op/erf.hpp
        op/exp.hpp
        op/expand.hpp
        op/expand.cpp
        op/eye_like.cpp
        op/eye_like.hpp
        op/flatten.cpp
        op/flatten.hpp
        op/floor.hpp
        op/gather.hpp
        op/gemm.cpp
        op/gemm.hpp
        op/global_average_pool.cpp
        op/global_average_pool.hpp
        op/global_max_pool.cpp
        op/global_max_pool.hpp
        op/greater.hpp
        op/hard_sigmoid.cpp
        op/hard_sigmoid.hpp
        op/hardmax.cpp
        op/hardmax.hpp
        op/identity.hpp
        op/instance_norm.cpp
        op/instance_norm.hpp
        op/leaky_relu.cpp
        op/leaky_relu.hpp
        op/less.hpp
        op/log.hpp
        op/log_softmax.hpp
        op/lp_norm.cpp
        op/lp_norm.hpp
        op/lp_pool.cpp
        op/lp_pool.hpp
        op/lrn.cpp
        op/lrn.hpp
        op/lstm.cpp
        op/lstm.hpp
        op/matmul.cpp
        op/matmul.hpp
        op/matmul_integer.cpp
        op/matmul_integer.hpp
        op/max_pool.cpp
        op/max_pool.hpp
        op/max.hpp
        op/mean.cpp
        op/mean.hpp
        op/mean_variance_normalization.cpp
        op/mean_variance_normalization.hpp
        op/min.hpp
        op/mul.hpp
        op/neg.hpp
        op/not.hpp
        op/onehot.cpp
        op/onehot.hpp
        op/or.hpp
        op/pad.cpp
        op/pad.hpp
        op/pow.hpp
        op/prelu.cpp
        op/prelu.hpp
        op/qlinear_matmul.cpp
        op/qlinear_matmul.hpp
        op/quant_conv.cpp
        op/quant_conv.hpp
        op/quantize_linear.cpp
        op/quantize_linear.hpp
        op/reciprocal.cpp
        op/reciprocal.hpp
        op/reduce.cpp
        op/reduce.hpp
        op/relu.hpp
        op/reshape.cpp
        op/reshape.hpp
        op/reverse_sequence.cpp
        op/reverse_sequence.h
        op/selu.cpp
        op/selu.hpp
        op/shape.hpp
        op/shape.cpp
        op/shrink.hpp
        op/shrink.cpp
        op/sigmoid.hpp
        op/sign.hpp
        op/sin.hpp
        op/sinh.hpp
        op/size.cpp
        op/size.hpp
        op/slice.cpp
        op/slice.hpp
        op/softmax.cpp
        op/softmax.hpp
        op/softplus.cpp
        op/softplus.hpp
        op/softsign.cpp
        op/softsign.hpp
        op/space_to_depth.cpp
        op/space_to_depth.hpp
        op/split.cpp
        op/split.hpp
        op/sqrt.hpp
        op/squeeze.cpp
        op/squeeze.hpp
        op/sub.hpp
        op/sum.hpp
        op/tan.hpp
        op/tanh.hpp
        op/thresholded_relu.cpp
        op/thresholded_relu.hpp
        op/topk.cpp
        op/topk.hpp
        op/transpose.cpp
        op/transpose.hpp
        op/unsqueeze.cpp
        op/unsqueeze.hpp
        op/where.hpp
        op/xor.hpp
        ops_bridge.cpp
        ops_bridge.hpp
        utils/common.cpp
        utils/common.hpp
        utils/convpool.cpp
        utils/convpool.hpp
        utils/pooling_factory.cpp
        utils/pooling_factory.hpp
        utils/reduction.cpp
        utils/reduction.hpp
        utils/reshape.cpp
        utils/reshape.hpp
        utils/variadic.hpp)

set(ONNX_IMPORT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

add_dependencies(onnx_import_interface protobuf::libprotobuf onnx::libonnx onnx::libonnx_proto)
add_dependencies(onnx_import onnx_import_interface)

set_property(TARGET onnx_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import 
    SYSTEM PRIVATE ${ONNX_IMPORT_INCLUDE_DIR} ${NGRAPH_INCLUDE_PATH}
    SYSTEM PRIVATE ${ONNX_INCLUDE_DIR} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIR})
target_link_libraries(onnx_import PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY})

target_compile_definitions(onnx_import PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

set_property(TARGET onnx_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import_interface SYSTEM PRIVATE ${ONNX_IMPORT_INCLUDE_DIR} ${NGRAPH_INCLUDE_PATH}
    SYSTEM PRIVATE ${ONNX_INCLUDE_DIR} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIR})

target_compile_definitions(onnx_import_interface PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
    target_compile_options(onnx_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()
