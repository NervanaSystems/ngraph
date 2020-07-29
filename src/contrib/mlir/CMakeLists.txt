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

if (NOT (NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE))
    return()
endif()

include_directories(
    ${NGRAPH_INCLUDE_PATH}
    ${MLIR_LLVM_INCLUDE_PATH}
    ${MLIR_INCLUDE_PATHS}
)

add_subdirectory(tools/ngraph-opt)

set(SRC
    backend/cpu/cpu_backend.cpp
    backend/pass/affine_lowerer.cpp
    backend/analysis/memory_analysis.cpp
    core/compiler.cpp
    core/ngraph_dialect/dialect.cpp
    core/ngraph_dialect/type.cpp
    core/ngraph_dialect/ops.cpp
    core/pass/mlir_subgraph_extraction.cpp
    core/pass/mlir_subgraph_extraction.hpp
    core/pass/ng_dialect_builder.cpp
    core/pass/ng_dialect_builder.hpp
    core/pass/ng_dialect_fused_ops.cpp
    core/pass/ng_dialect_fused_ops.hpp
    core/pass/ng_op_fusion.cpp
    core/pass/ng_op_fusion.hpp
    core/pass/convert/abs_v0.cpp
    core/pass/convert/acosh_v3.cpp
    core/pass/convert/acos_v0.cpp
    core/pass/convert/add_v0.cpp
    core/pass/convert/add_v1.cpp
    core/pass/convert/all_reduce_v0.cpp
    core/pass/convert/all_v0.cpp
    core/pass/convert/and_v0.cpp
    core/pass/convert/any_v0.cpp
    core/pass/convert/arg_max_v0.cpp
    core/pass/convert/arg_min_v0.cpp
    core/pass/convert/asinh_v3.cpp
    core/pass/convert/asin_v0.cpp
    core/pass/convert/atan2_v0.cpp
    core/pass/convert/atanh_v3.cpp
    core/pass/convert/atan_v0.cpp
    core/pass/convert/avg_pool_backprop_v0.cpp
    core/pass/convert/avg_pool_v0.cpp
    core/pass/convert/avg_pool_v1.cpp
    core/pass/convert/batch_mat_mul_transpose_v0.cpp
    core/pass/convert/batch_mat_mul_v0.cpp
    core/pass/convert/batch_norm_inference_v0.cpp
    core/pass/convert/batch_norm_training_backprop_v0.cpp
    core/pass/convert/batch_norm_training_v0.cpp
    core/pass/convert/batch_to_space_v1.cpp
    core/pass/convert/binary_convolution_v1.cpp
    core/pass/convert/broadcast_distributed_v0.cpp
    core/pass/convert/broadcast_like_v0.cpp
    core/pass/convert/broadcast_v0.cpp
    core/pass/convert/broadcast_v1.cpp
    core/pass/convert/broadcast_v3.cpp
    core/pass/convert/bucketize_v3.cpp
    core/pass/convert/ceiling_v0.cpp
    core/pass/convert/clamp_v0.cpp
    core/pass/convert/concat_v0.cpp
    core/pass/convert/constant_v0.cpp
    core/pass/convert/convert_like_v1.cpp
    core/pass/convert/convert_v0.cpp
    core/pass/convert/convolution_backprop_data_v0.cpp
    core/pass/convert/convolution_backprop_data_v1.cpp
    core/pass/convert/convolution_backprop_filters_v0.cpp
    core/pass/convert/convolution_bias_add_v0.cpp
    core/pass/convert/convolution_bias_backprop_filters_bias_v0.cpp
    core/pass/convert/convolution_bias_v0.cpp
    core/pass/convert/convolution_v0.cpp
    core/pass/convert/convolution_v1.cpp
    core/pass/convert/cosh_v0.cpp
    core/pass/convert/cos_v0.cpp
    core/pass/convert/crop_and_resize_v0.cpp
    core/pass/convert/cross_entropy_backprop_v0.cpp
    core/pass/convert/cross_entropy_v0.cpp
    core/pass/convert/ctc_greedy_decoder_v0.cpp
    core/pass/convert/cum_sum_v0.cpp
    core/pass/convert/deformable_convolution_v1.cpp
    core/pass/convert/deformablepsroi_pooling_v1.cpp
    core/pass/convert/depth_to_space_v0.cpp
    core/pass/convert/dequantize_v0.cpp
    core/pass/convert/detection_output_v0.cpp
    core/pass/convert/divide_v0.cpp
    core/pass/convert/divide_v1.cpp
    core/pass/convert/dot_v0.cpp
    core/pass/convert/dyn_broadcast_v0.cpp
    core/pass/convert/dyn_pad_v0.cpp
    core/pass/convert/dyn_replace_slice_v0.cpp
    core/pass/convert/dyn_slice_v0.cpp
    core/pass/convert/elu_v0.cpp
    core/pass/convert/embedding_bag_offsets_sum_v3.cpp
    core/pass/convert/embedding_bag_packed_sum_v3.cpp
    core/pass/convert/embedding_lookup_v0.cpp
    core/pass/convert/embedding_segments_sum_v3.cpp
    core/pass/convert/equal_v0.cpp
    core/pass/convert/equal_v1.cpp
    core/pass/convert/erf_v0.cpp
    core/pass/convert/exp_v0.cpp
    core/pass/convert/extract_image_patches_v3.cpp
    core/pass/convert/fake_quantize_v0.cpp
    core/pass/convert/floor_mod_v1.cpp
    core/pass/convert/floor_v0.cpp
    core/pass/convert/gathernd_v0.cpp
    core/pass/convert/gather_tree_v1.cpp
    core/pass/convert/gather_v0.cpp
    core/pass/convert/gather_v1.cpp
    core/pass/convert/gelu_backprop_factor_v0.cpp
    core/pass/convert/gelu_v0.cpp
    core/pass/convert/gemm_v0.cpp
    core/pass/convert/generate_mask_v0.cpp
    core/pass/convert/greater_equal_v1.cpp
    core/pass/convert/greater_eq_v0.cpp
    core/pass/convert/greater_v0.cpp
    core/pass/convert/greater_v1.cpp
    core/pass/convert/grn_v0.cpp
    core/pass/convert/group_convolution_backprop_data_v0.cpp
    core/pass/convert/group_convolution_backprop_data_v1.cpp
    core/pass/convert/group_convolution_backprop_filters_v0.cpp
    core/pass/convert/group_convolution_v0.cpp
    core/pass/convert/group_convolution_v1.cpp
    core/pass/convert/gru_cell_v3.cpp
    core/pass/convert/hard_sigmoid_v0.cpp
    core/pass/convert/interpolate_v0.cpp
    core/pass/convert/interpolate_v3.cpp
    core/pass/convert/layer_norm_backprop_v0.cpp
    core/pass/convert/layer_norm_v0.cpp
    core/pass/convert/less_equal_v1.cpp
    core/pass/convert/less_eq_v0.cpp
    core/pass/convert/less_v0.cpp
    core/pass/convert/less_v1.cpp
    core/pass/convert/logical_and_v1.cpp
    core/pass/convert/logical_not_v1.cpp
    core/pass/convert/logical_or_v1.cpp
    core/pass/convert/logical_xor_v1.cpp
    core/pass/convert/log_v0.cpp
    core/pass/convert/lrn_v0.cpp
    core/pass/convert/lstm_cell_v0.cpp
    core/pass/convert/lstm_sequence_v0.cpp
    core/pass/convert/mat_mul_v0.cpp
    core/pass/convert/maximum_v0.cpp
    core/pass/convert/maximum_v1.cpp
    core/pass/convert/max_pool_backprop_v0.cpp
    core/pass/convert/max_pool_v0.cpp
    core/pass/convert/max_pool_v1.cpp
    core/pass/convert/max_v0.cpp
    core/pass/convert/minimum_v0.cpp
    core/pass/convert/minimum_v1.cpp
    core/pass/convert/min_v0.cpp
    core/pass/convert/mod_v1.cpp
    core/pass/convert/multiply_v0.cpp
    core/pass/convert/multiply_v1.cpp
    core/pass/convert/mvn_v0.cpp
    core/pass/convert/negative_v0.cpp
    core/pass/convert/non_max_suppression_v1.cpp
    core/pass/convert/non_max_suppression_v3.cpp
    core/pass/convert/non_zero_v3.cpp
    core/pass/convert/normalize_l2_v0.cpp
    core/pass/convert/not_equal_v0.cpp
    core/pass/convert/not_equal_v1.cpp
    core/pass/convert/not_v0.cpp
    core/pass/convert/one_hot_v0.cpp
    core/pass/convert/one_hot_v1.cpp
    core/pass/convert/or_v0.cpp
    core/pass/convert/pad_v0.cpp
    core/pass/convert/pad_v1.cpp
    core/pass/convert/parameter_v0.cpp
    core/pass/convert/partial_slice_backprop_v0.cpp
    core/pass/convert/partial_slice_v0.cpp
    core/pass/convert/passthrough_v0.cpp
    core/pass/convert/power_v0.cpp
    core/pass/convert/power_v1.cpp
    core/pass/convert/p_relu_v0.cpp
    core/pass/convert/prior_box_clustered_v0.cpp
    core/pass/convert/prior_box_v0.cpp
    core/pass/convert/product_v0.cpp
    core/pass/convert/proposal_v0.cpp
    core/pass/convert/psroi_pooling_v0.cpp
    core/pass/convert/quantized_convolution_bias_add_v0.cpp
    core/pass/convert/quantized_convolution_bias_signed_add_v0.cpp
    core/pass/convert/quantized_convolution_bias_v0.cpp
    core/pass/convert/quantized_convolution_relu_v0.cpp
    core/pass/convert/quantized_convolution_v0.cpp
    core/pass/convert/quantized_dot_bias_v0.cpp
    core/pass/convert/quantized_dot_v0.cpp
    core/pass/convert/quantize_v0.cpp
    core/pass/convert/random_uniform_v0.cpp
    core/pass/convert/range_v0.cpp
    core/pass/convert/recv_v0.cpp
    core/pass/convert/reduce_logical_and_v1.cpp
    core/pass/convert/reduce_logical_or_v1.cpp
    core/pass/convert/reduce_max_v1.cpp
    core/pass/convert/reduce_mean_v1.cpp
    core/pass/convert/reduce_min_v1.cpp
    core/pass/convert/reduce_prod_v1.cpp
    core/pass/convert/reduce_sum_v1.cpp
    core/pass/convert/region_yolo_v0.cpp
    core/pass/convert/relu_backprop_v0.cpp
    core/pass/convert/relu_v0.cpp
    core/pass/convert/reorg_yolo_v0.cpp
    core/pass/convert/replace_slice_v0.cpp
    core/pass/convert/reshape_v0.cpp
    core/pass/convert/reshape_v1.cpp
    core/pass/convert/result_v0.cpp
    core/pass/convert/reverse_sequence_v0.cpp
    core/pass/convert/reverse_v0.cpp
    core/pass/convert/reverse_v1.cpp
    core/pass/convert/rnn_cell_v0.cpp
    core/pass/convert/roi_align_v3.cpp
    core/pass/convert/roi_pooling_v0.cpp
    core/pass/convert/round_v0.cpp
    core/pass/convert/scalar_constant_like_v0.cpp
    core/pass/convert/scale_shift_v0.cpp
    core/pass/convert/scatter_add_v0.cpp
    core/pass/convert/scatter_elements_update_v3.cpp
    core/pass/convert/scatternd_add_v0.cpp
    core/pass/convert/scatternd_v0.cpp
    core/pass/convert/scatter_update_v3.cpp
    core/pass/convert/select_v0.cpp
    core/pass/convert/select_v1.cpp
    core/pass/convert/selu_v0.cpp
    core/pass/convert/send_v0.cpp
    core/pass/convert/shape_of_v0.cpp
    core/pass/convert/shape_of_v3.cpp
    core/pass/convert/shuffle_channels_v0.cpp
    core/pass/convert/sigmoid_backprop_v0.cpp
    core/pass/convert/sigmoid_v0.cpp
    core/pass/convert/sign_v0.cpp
    core/pass/convert/sinh_v0.cpp
    core/pass/convert/sin_v0.cpp
    core/pass/convert/slice_v0.cpp
    core/pass/convert/softmax_cross_entropy_backprop_v0.cpp
    core/pass/convert/softmax_cross_entropy_v0.cpp
    core/pass/convert/softmax_v0.cpp
    core/pass/convert/softmax_v1.cpp
    core/pass/convert/space_to_batch_v1.cpp
    core/pass/convert/space_to_depth_v0.cpp
    core/pass/convert/split_v0.cpp
    core/pass/convert/split_v1.cpp
    core/pass/convert/sqrt_v0.cpp
    core/pass/convert/squared_difference_v0.cpp
    core/pass/convert/squeeze_v0.cpp
    core/pass/convert/stack_v0.cpp
    core/pass/convert/stop_gradient_v0.cpp
    core/pass/convert/strided_slice_v1.cpp
    core/pass/convert/subtract_v0.cpp
    core/pass/convert/subtract_v1.cpp
    core/pass/convert/sum_v0.cpp
    core/pass/convert/tanh_v0.cpp
    core/pass/convert/tan_v0.cpp
    core/pass/convert/tensor_iterator_v0.cpp
    core/pass/convert/tile_v0.cpp
    core/pass/convert/topk_v0.cpp
    core/pass/convert/topk_v1.cpp
    core/pass/convert/topk_v3.cpp
    core/pass/convert/transpose_v1.cpp
    core/pass/convert/unsqueeze_v0.cpp
    core/pass/convert/variadic_split_v1.cpp
    core/pass/convert/xor_v0.cpp
    runtime/cpu/memory_manager.cpp
    runtime/cpu/cpu_runtime.cpp
    runtime/cpu/cpu_callbacks.cpp
    utils.cpp
)

add_library(mlir_llvm_backend SHARED ${SRC})

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link MLIR libs
target_link_libraries(
    mlir_llvm_backend PRIVATE
    MLIRAffineOps
    MLIRAffineEDSC
    MLIRAffineToStandard
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRSCFToStandard
    MLIRStandardToLLVM
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRSupport
    MLIRAffineTransforms
)
# some libs need whole archive linkage because of Globals static initialization
function(whole_archive_link target)
    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        set(link_flags "-L${LLVM_BUILD_LIBRARY_DIR} -Wl,-force_load ")
        FOREACH(LIB ${ARGN})
            string(CONCAT link_flags ${link_flags} "${LIB} ")
        ENDFOREACH(LIB)
    else()
        set(link_flags "-Llib -Wl,--whole-archive,")
        FOREACH(LIB ${ARGN})
            string(CONCAT link_flags ${link_flags} "${LIB},")
        ENDFOREACH(LIB)
        string(CONCAT link_flags ${link_flags} "--no-whole-archive")
    endif()
    message(STATUS "MLIR Ops link flag: ${link_flags}" )

    set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

set(LIBS
    ${LLVM_BUILD_LIBRARY_DIR}/libMLIRAffineOps.a
    ${LLVM_BUILD_LIBRARY_DIR}/libMLIRStandardOps.a
)
whole_archive_link(mlir_llvm_backend ${LIBS})
# Link LLVM libs
target_link_libraries(
    mlir_llvm_backend PRIVATE
    ${llvm_libs}
)

# Link ngraph
target_link_libraries(mlir_llvm_backend PUBLIC ngraph libmkl DNNL::dnnl)

# table-gen dialect ops
# include table-gen helpers
include(${LLVM_DIR}/TableGen.cmake)

function(ngraph_tablegen ofn)
    tablegen(MLIR ${ARGV} "-I${MLIR_SRC_INCLUDE_PATH}" "-I${MLIR_BIN_INCLUDE_PATH}")
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
endfunction()


set(MLIR_TABLEGEN_EXE mlir-tblgen)

# table-gen ops.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops.td)
ngraph_tablegen(ops.h.inc -gen-op-decls)
ngraph_tablegen(ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ngraph_ops_gen)

# table-gen ops_interfaces.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops_interfaces.td)
ngraph_tablegen(ops_interfaces.h.inc -gen-op-interface-decls)
ngraph_tablegen(ops_interfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ngraph_ops_interfaces_gen)

# tabel-gen ops attributes.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops_attributes.td)
ngraph_tablegen(ops_attributes.h.inc -gen-enum-decls)
ngraph_tablegen(ops_attributes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(ngraph_ops_attributes_gen)

# tabel-gen ops fused_ops_pattern.td
set(LLVM_TARGET_DEFINITIONS core/pass/fused_ops_pattern.td)
ngraph_tablegen(fused_ops_pattern.h.inc -gen-rewriters)
add_public_tablegen_target(ngraph_ops_pattern_gen)

add_dependencies(mlir_llvm_backend ngraph_ops_gen ngraph_ops_interfaces_gen ngraph_ops_attributes_gen ngraph_ops_pattern_gen)

target_include_directories(mlir_llvm_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS mlir_llvm_backend DESTINATION ${NGRAPH_INSTALL_LIB})
