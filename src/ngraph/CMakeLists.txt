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

set (SRC
    assertion.cpp
    assertion.hpp
    attribute_adapter.cpp
    attribute_adapter.hpp
    attribute_visitor.cpp
    attribute_visitor.hpp
    autodiff/adjoints.cpp
    autodiff/adjoints.hpp
    axis_set.cpp
    axis_set.hpp
    axis_vector.cpp
    axis_vector.hpp
    builder/autobroadcast.cpp
    builder/autobroadcast.hpp
    builder/dequantize_builder.cpp
    builder/dequantize_builder.hpp
    builder/make_constant.hpp
    builder/matmul_factory.cpp
    builder/matmul_factory.hpp
    builder/norm.cpp
    builder/norm.hpp
    builder/numpy_transpose.cpp
    builder/numpy_transpose.hpp
    builder/quantization_utils.cpp
    builder/quantization_utils.hpp
    builder/quantization/quantized_linear_convolution.cpp
    builder/quantization/quantized_linear_convolution.hpp
    builder/quantize_builder.cpp
    builder/quantize_builder.hpp
    builder/quantized_concat_builder.cpp
    builder/quantized_concat_builder.hpp
    builder/quantized_conv_builder.cpp
    builder/quantized_conv_builder.hpp
    builder/quantized_dot_builder.cpp
    builder/quantized_dot_builder.hpp
    builder/reduce_ops.cpp
    builder/reduce_ops.hpp
    builder/reshape.cpp
    builder/reshape.hpp
    builder/split.cpp
    builder/split.hpp
    builder/tensor_mask.hpp
    check.hpp
    chrome_trace.cpp
    chrome_trace.hpp
    code_writer.hpp
    coordinate_diff.cpp
    coordinate_diff.hpp
    coordinate_transform.cpp
    coordinate.cpp
    coordinate.hpp
    cpio.cpp
    cpio.hpp
    deprecated.hpp
    descriptor/input.cpp
    descriptor/input.hpp
    descriptor/layout/dense_tensor_layout.cpp
    descriptor/layout/dense_tensor_layout.hpp
    descriptor/layout/tensor_layout.cpp
    descriptor/layout/tensor_layout.hpp
    descriptor/output.cpp
    descriptor/output.hpp
    descriptor/tensor.cpp
    descriptor/tensor.hpp
    dimension.cpp
    dimension.hpp
    distributed.cpp
    distributed.hpp
    distributed/null.cpp
    distributed/null.hpp
    enum_names.hpp
    env_util.cpp
    env_util.hpp
    evaluator.hpp
    except.hpp
    factory_adapter.hpp
    factory.cpp
    factory.hpp
    file_util.cpp
    file_util.hpp
    function.cpp
    function.hpp
    graph_util.cpp
    interval.cpp
    interval.hpp
    lambda.cpp
    lambda.hpp
    log.cpp
    log.hpp
    ngraph_visibility.hpp
    ngraph.cpp
    ngraph.hpp
    node_input.cpp
    node_input.hpp
    node_output.cpp
    node_output.hpp
    node.cpp
    node.hpp
    op/abs.cpp
    op/abs.hpp
    op/acos.cpp
    op/acos.hpp
    op/acosh.cpp
    op/acosh.hpp
    op/add.cpp
    op/add.hpp
    op/all.cpp
    op/all.hpp
    op/allreduce.cpp
    op/allreduce.hpp
    op/any.cpp
    op/any.hpp
    op/argmax.cpp
    op/argmax.hpp
    op/argmin.cpp
    op/argmin.hpp
    op/asin.cpp
    op/asin.hpp
    op/asinh.cpp
    op/asinh.hpp
    op/atan.cpp
    op/atan.hpp
    op/atan2.cpp
    op/atan2.hpp
    op/atanh.cpp
    op/atanh.hpp
    op/avg_pool.cpp
    op/avg_pool.hpp
    op/batch_mat_mul_transpose.cpp
    op/batch_mat_mul_transpose.hpp
    op/batch_norm.cpp
    op/batch_norm.hpp
    op/batch_to_space.cpp
    op/batch_to_space.hpp
    op/binary_convolution.cpp
    op/binary_convolution.hpp
    op/broadcast_distributed.cpp
    op/broadcast.cpp
    op/bucketize.cpp
    op/bucketize.hpp
    op/ceiling.cpp
    op/ceiling.hpp
    op/clamp.cpp
    op/clamp.hpp
    op/concat.cpp
    op/concat.hpp
    op/constant.cpp
    op/constant.hpp
    op/conv_fused.cpp
    op/conv_fused.hpp
    op/convert_like.cpp
    op/convert_like.hpp
    op/convert.cpp
    op/convert.hpp
    op/convolution.cpp
    op/convolution.hpp
    op/cos.cpp
    op/cos.hpp
    op/cosh.cpp
    op/cosh.hpp
    op/crop_and_resize.cpp
    op/crop_and_resize.hpp
    op/crossentropy.cpp
    op/crossentropy.hpp
    op/ctc_greedy_decoder.cpp
    op/ctc_greedy_decoder.hpp
    op/cum_sum.cpp
    op/cum_sum.hpp
    op/deformable_convolution.cpp
    op/deformable_convolution.hpp
    op/deformable_psroi_pooling.cpp
    op/deformable_psroi_pooling.hpp
    op/depth_to_space.cpp
    op/depth_to_space.hpp
    op/dequantize.cpp
    op/dequantize.hpp
    op/detection_output.cpp
    op/detection_output.hpp
    op/divide.cpp
    op/divide.hpp
    op/dot.cpp
    op/dot.hpp
    op/elu.cpp
    op/elu.hpp
    op/embedding_lookup.cpp
    op/embedding_lookup.hpp
    op/embedding_segments_sum.cpp
    op/embedding_segments_sum.hpp
    op/embeddingbag_offsets_sum.cpp
    op/embeddingbag_offsets_sum.hpp
    op/embeddingbag_packedsum.cpp
    op/embeddingbag_packedsum.hpp
    op/equal.cpp
    op/equal.hpp
    op/erf.cpp
    op/erf.hpp
    op/exp.cpp
    op/exp.hpp
    op/experimental/batch_mat_mul.cpp
    op/experimental/batch_mat_mul.hpp
    op/experimental/compiled_kernel.cpp
    op/experimental/compiled_kernel.hpp
    op/experimental/dyn_broadcast.cpp
    op/experimental/dyn_broadcast.hpp
    op/experimental/dyn_pad.cpp
    op/experimental/dyn_pad.hpp
    op/experimental/dyn_replace_slice.cpp
    op/experimental/dyn_replace_slice.hpp
    op/experimental/dyn_slice.cpp
    op/experimental/dyn_slice.hpp
    op/experimental/generate_mask.cpp
    op/experimental/generate_mask.hpp
    op/experimental/layers/ctc_greedy_decoder.hpp
    op/experimental/layers/detection_output.hpp
    op/experimental/layers/interpolate.hpp
    op/experimental/layers/prior_box_clustered.hpp
    op/experimental/layers/prior_box.hpp
    op/experimental/layers/proposal.hpp
    op/experimental/layers/psroi_pooling.hpp
    op/experimental/layers/region_yolo.hpp
    op/experimental/layers/reorg_yolo.hpp
    op/experimental/layers/roi_pooling.hpp
    op/experimental/quantized_conv_bias.cpp
    op/experimental/quantized_conv_bias.hpp
    op/experimental/quantized_conv_relu.cpp
    op/experimental/quantized_conv_relu.hpp
    op/experimental/quantized_dot_bias.cpp
    op/experimental/quantized_dot_bias.hpp
    op/experimental/random_uniform.cpp
    op/experimental/random_uniform.hpp
    op/experimental/range.hpp
    op/experimental/shape_of.hpp
    op/experimental/tile.hpp
    op/experimental/transpose.hpp
    op/extractimagepatches.cpp
    op/extractimagepatches.hpp
    op/fake_quantize.cpp
    op/fake_quantize.hpp
    op/floor_mod.cpp
    op/floor_mod.hpp
    op/floor.cpp
    op/floor.hpp
    op/gather_nd.cpp
    op/gather_nd.hpp
    op/gather_tree.cpp
    op/gather_tree.hpp
    op/gather.cpp
    op/gather.hpp
    op/gelu.cpp
    op/gelu.hpp
    op/gemm.cpp
    op/gemm.hpp
    op/greater_equal.cpp
    op/greater_equal.hpp
    op/greater.cpp
    op/greater.hpp
    op/grn.cpp
    op/grn.hpp
    op/group_conv.cpp
    op/group_conv.hpp
    op/group_conv.hpp
    op/gru_cell.cpp
    op/gru_cell.hpp
    op/hard_sigmoid.cpp
    op/hard_sigmoid.hpp
    op/interpolate.cpp
    op/interpolate.hpp
    op/layer_norm.cpp
    op/layer_norm.hpp
    op/less_equal.cpp
    op/less_equal.hpp
    op/less.cpp
    op/less.hpp
    op/log.cpp
    op/log.hpp
    op/logical_and.cpp
    op/logical_and.hpp
    op/logical_not.cpp
    op/logical_not.hpp
    op/logical_or.cpp
    op/logical_or.hpp
    op/logical_xor.cpp
    op/logical_xor.hpp
    op/lrn.cpp
    op/lrn.hpp
    op/lstm_cell.cpp
    op/lstm_cell.hpp
    op/lstm_sequence.cpp
    op/lstm_sequence.hpp
    op/matmul.cpp
    op/matmul.hpp
    op/max_pool.cpp
    op/max_pool.hpp
    op/max.cpp
    op/max.hpp
    op/maximum.cpp
    op/maximum.hpp
    op/min.cpp
    op/min.hpp
    op/minimum.cpp
    op/minimum.hpp
    op/mod.cpp
    op/mod.hpp
    op/multiply.cpp
    op/multiply.hpp
    op/mvn.cpp
    op/mvn.hpp
    op/negative.cpp
    op/negative.hpp
    op/non_max_suppression.cpp
    op/non_max_suppression.hpp
    op/non_zero.cpp
    op/non_zero.hpp
    op/normalize_l2.cpp
    op/normalize_l2.hpp
    op/not_equal.cpp
    op/not_equal.hpp
    op/one_hot.cpp
    op/one_hot.hpp
    op/op.cpp
    op/op.hpp
    op/pad.cpp
    op/pad.hpp
    op/parameter.cpp
    op/parameter.hpp
    op/partial_slice.cpp
    op/partial_slice.hpp
    op/passthrough.cpp
    op/passthrough.hpp
    op/power.cpp
    op/power.hpp
    op/prelu.cpp
    op/prelu.hpp
    op/prior_box_clustered.cpp
    op/prior_box_clustered.hpp
    op/prior_box.cpp
    op/prior_box.hpp
    op/product.cpp
    op/product.hpp
    op/proposal.cpp
    op/proposal.hpp
    op/psroi_pooling.cpp
    op/psroi_pooling.hpp
    op/quantize.cpp
    op/quantize.hpp
    op/quantized_convolution.cpp
    op/quantized_convolution.hpp
    op/quantized_dot.cpp
    op/quantized_dot.hpp
    op/range.cpp
    op/range.hpp
    op/recv.cpp
    op/recv.hpp
    op/reduce_logical_and.cpp
    op/reduce_logical_and.hpp
    op/reduce_logical_or.cpp
    op/reduce_logical_or.hpp
    op/reduce_mean.cpp
    op/reduce_mean.hpp
    op/reduce_prod.cpp
    op/reduce_prod.hpp
    op/reduce_sum.cpp
    op/reduce_sum.hpp
    op/region_yolo.cpp
    op/region_yolo.hpp
    op/relu.cpp
    op/relu.hpp
    op/reorg_yolo.cpp
    op/reorg_yolo.hpp
    op/replace_slice.cpp
    op/replace_slice.hpp
    op/reshape.cpp
    op/reshape.hpp
    op/result.cpp
    op/result.hpp
    op/reverse_sequence.cpp
    op/reverse_sequence.hpp
    op/reverse.cpp
    op/reverse.hpp
    op/rnn_cell.cpp
    op/rnn_cell.hpp
    op/roi_align.cpp
    op/roi_align.hpp
    op/roi_pooling.cpp
    op/roi_pooling.hpp
    op/round.cpp
    op/round.hpp
    op/scale_shift.cpp
    op/scale_shift.hpp
    op/scatter_add.cpp
    op/scatter_add.hpp
    op/scatter_elements_update.cpp
    op/scatter_elements_update.hpp
    op/scatter_nd_add.cpp
    op/scatter_nd_add.hpp
    op/scatter_nd_update.cpp
    op/scatter_nd_update.hpp
    op/scatter_nd.cpp
    op/scatter_nd.hpp
    op/scatter_update.cpp
    op/scatter_update.hpp
    op/select.cpp
    op/select.hpp
    op/selu.cpp
    op/selu.hpp
    op/send.cpp
    op/send.hpp
    op/shape_of.cpp
    op/shape_of.hpp
    op/shuffle_channels.cpp
    op/shuffle_channels.hpp
    op/sigmoid.cpp
    op/sigmoid.hpp
    op/sign.cpp
    op/sign.hpp
    op/sin.cpp
    op/sin.hpp
    op/sinh.cpp
    op/sinh.hpp
    op/slice.cpp
    op/slice.hpp
    op/softmax_crossentropy.cpp
    op/softmax_crossentropy.hpp
    op/softmax.cpp
    op/softmax.hpp
    op/space_to_batch.cpp
    op/space_to_batch.hpp
    op/space_to_depth.cpp
    op/space_to_depth.hpp
    op/split.cpp
    op/split.hpp
    op/split.hpp
    op/sqrt.cpp
    op/sqrt.hpp
    op/squared_difference.cpp
    op/squared_difference.hpp
    op/squeeze.cpp
    op/squeeze.hpp
    op/stack.cpp
    op/stack.hpp
    op/stop_gradient.cpp
    op/stop_gradient.hpp
    op/strided_slice.cpp
    op/strided_slice.hpp
    op/strided_slice.hpp
    op/subtract.cpp
    op/subtract.hpp
    op/sum.cpp
    op/sum.hpp
    op/tan.cpp
    op/tan.hpp
    op/tanh.cpp
    op/tanh.hpp
    op/tensor_iterator.cpp
    op/tensor_iterator.hpp
    op/tile.cpp
    op/tile.hpp
    op/topk.cpp
    op/topk.hpp
    op/transpose.cpp
    op/transpose.hpp
    op/unsqueeze.cpp
    op/unsqueeze.hpp
    op/util/activation_functions.cpp
    op/util/activation_functions.hpp
    op/util/arithmetic_reduction.cpp
    op/util/arithmetic_reduction.hpp
    op/util/arithmetic_reductions_keep_dims.cpp
    op/util/arithmetic_reductions_keep_dims.hpp
    op/util/attr_types.cpp
    op/util/attr_types.hpp
    op/util/binary_elementwise_arithmetic.cpp
    op/util/binary_elementwise_arithmetic.hpp
    op/util/binary_elementwise_comparison.cpp
    op/util/binary_elementwise_comparison.hpp
    op/util/binary_elementwise_logical.cpp
    op/util/binary_elementwise_logical.hpp
    op/util/broadcast_base.cpp
    op/util/broadcast_base.hpp
    op/util/embeddingbag_offsets_base.cpp
    op/util/embeddingbag_offsets_base.hpp
    op/util/embeddingbag_packed_base.cpp
    op/util/embeddingbag_packed_base.hpp
    op/util/fused_op.cpp
    op/util/fused_op.hpp
    op/util/index_reduction.cpp
    op/util/index_reduction.hpp
    op/util/logical_reduction_keep_dims.cpp
    op/util/logical_reduction_keep_dims.hpp
    op/util/logical_reduction.cpp
    op/util/logical_reduction.hpp
    op/util/rnn_cell_base.cpp
    op/util/rnn_cell_base.hpp
    op/util/scatter_base.cpp
    op/util/scatter_base.hpp
    op/util/scatter_nd_base.cpp
    op/util/scatter_nd_base.hpp
    op/util/unary_elementwise_arithmetic.cpp
    op/util/unary_elementwise_arithmetic.hpp
    op/variadic_split.cpp
    op/variadic_split.hpp
    ops.hpp
    opset/opset.cpp
    partial_shape.cpp
    partial_shape.hpp
    pass/algebraic_simplification.cpp
    pass/algebraic_simplification.hpp
    pass/assign_layout.hpp
    pass/batch_fusion.cpp
    pass/batch_fusion.hpp
    pass/common_function_collection.cpp
    pass/common_function_collection.hpp
    pass/concat_fusion.cpp
    pass/concat_fusion.hpp
    pass/constant_folding_arithmetic_reduction.cpp
    pass/constant_folding_convert.cpp
    pass/constant_folding_dequantize.cpp
    pass/constant_folding_dyn_broadcast.cpp
    pass/constant_folding_dyn_reshape.cpp
    pass/constant_folding_dyn_slice.cpp
    pass/constant_folding_gather.cpp
    pass/constant_folding_logical_reduction.cpp
    pass/constant_folding_one_hot.cpp
    pass/constant_folding_pad.cpp
    pass/constant_folding_quantize.cpp
    pass/constant_folding_reverse.cpp
    pass/constant_folding_scatter.cpp
    pass/constant_folding_select.cpp
    pass/constant_folding_slice.cpp
    pass/constant_folding_split.cpp
    pass/constant_folding_tile.cpp
    pass/constant_folding_transpose.cpp
    pass/constant_folding_variadic_split.cpp
    pass/constant_folding.cpp
    pass/constant_folding.hpp
    pass/constant_to_broadcast.cpp
    pass/convert_fp32_to_fp16.cpp
    pass/convert_fp32_to_fp16.hpp
    pass/core_fusion.cpp
    pass/core_fusion.hpp
    pass/cse.cpp
    pass/cse.hpp
    pass/dump_sorted.cpp
    pass/dump_sorted.hpp
    pass/dyn_elimination.cpp
    pass/dyn_elimination.hpp
    pass/fused_op_decomposition.cpp
    pass/fused_op_decomposition.hpp
    pass/graph_rewrite.cpp
    pass/graph_rewrite.hpp
    pass/implicit_broadcast_elimination.cpp
    pass/implicit_broadcast_elimination.hpp
    pass/like_replacement.cpp
    pass/like_replacement.hpp
    pass/liveness.cpp
    pass/liveness.hpp
    pass/manager_state.hpp
    pass/manager.cpp
    pass/manager.hpp
    pass/memory_layout.cpp
    pass/memory_layout.hpp
    pass/memory_visualize.cpp
    pass/memory_visualize.hpp
    pass/nop_elimination.cpp
    pass/nop_elimination.hpp
    pass/convert_opset_0_to_1.cpp
    pass/convert_opset_0_to_1.hpp
    pass/convert_opset_1_to_0.cpp
    pass/convert_opset_1_to_0.hpp
    pass/convert_opset_3_to_1.cpp
    pass/convert_opset_3_to_1.hpp
    pass/pass_config.cpp
    pass/pass_config.hpp
    pass/pass_util.cpp
    pass/pass_util.hpp
    pass/pass.cpp
    pass/pass.hpp
    pass/propagate_cacheability.cpp
    pass/propagate_cacheability.hpp
    pass/reshape_elimination_v1.cpp
    pass/reshape_elimination_v1.hpp
    pass/reshape_elimination.cpp
    pass/reshape_elimination.hpp
    pass/reshape_sinking.cpp
    pass/reshape_sinking.hpp
    pass/serialize.cpp
    pass/serialize.hpp
    pass/shape_relevance.cpp
    pass/shape_relevance.hpp
    pass/validate_graph.cpp
    pass/validate_graph.hpp
    pass/validate.cpp
    pass/validate.hpp
    pass/visualize_tree.cpp
    pass/visualize_tree.hpp
    pass/zero_dim_tensor_elimination.cpp
    pass/zero_dim_tensor_elimination.cpp
    pass/zero_dim_tensor_elimination.hpp
    pass/zero_dim_tensor_elimination.hpp
    pattern/matcher.cpp
    pattern/matcher.hpp
    pattern/op/any_of.cpp
    pattern/op/any_of.hpp
    pattern/op/any.cpp
    pattern/op/any.hpp
    pattern/op/branch.cpp
    pattern/op/branch.hpp
    pattern/op/capture.cpp
    pattern/op/capture.hpp
    pattern/op/label.cpp
    pattern/op/label.hpp
    pattern/op/or.cpp
    pattern/op/or.hpp
    pattern/op/pattern.cpp
    pattern/op/pattern.hpp
    pattern/op/skip.cpp
    pattern/op/skip.hpp
    pattern/op/true.cpp
    pattern/op/true.hpp
    provenance.cpp
    provenance.hpp
    rank.hpp
    rt_info.cpp
    rt_info.hpp
    runtime/aligned_buffer.cpp
    runtime/aligned_buffer.hpp
    runtime/allocator.cpp
    runtime/allocator.hpp
    runtime/backend_manager.cpp
    runtime/backend_manager.hpp
    runtime/backend.cpp
    runtime/backend.hpp
    runtime/executable_cache.cpp
    runtime/executable_cache.hpp
    runtime/executable.cpp
    runtime/executable.hpp
    runtime/host_tensor.cpp
    runtime/host_tensor.hpp
    runtime/performance_counter.hpp
    runtime/tensor.cpp
    runtime/tensor.hpp
    shape_util.cpp
    shape_util.hpp
    shape.cpp
    shape.hpp
    slice_plan.cpp
    slice_plan.hpp
    specialize_function.cpp
    specialize_function.hpp
    state/bernoulli_rng_state.cpp
    state/bernoulli_rng_state.hpp
    state/uniform_rng_state.cpp
    state/uniform_rng_state.hpp
    strides.cpp
    strides.hpp
    type.cpp
    type.hpp
    type/bfloat16.cpp
    type/bfloat16.hpp
    type/element_type_traits.hpp
    type/element_type.cpp
    type/float16.cpp
    type/float16.hpp
    util.cpp
    util.hpp
    validation_util.cpp
    validation_util.hpp
    variant.cpp
    variant.hpp
)

set(SRC ${SRC}
    runtime/dynamic/dynamic_backend.cpp
    runtime/dynamic/dynamic_backend.hpp
    runtime/dynamic/dynamic_executable.cpp
    runtime/dynamic/dynamic_executable.hpp
    runtime/dynamic/dynamic_tensor.cpp
    runtime/dynamic/dynamic_tensor.hpp
)

# generate the op_version_tbl.hpp file used by the Node Factory
set(OPSETS
    opset/opset0_tbl.hpp
    opset/opset1_tbl.hpp
    opset/opset3_tbl.hpp
    opset/opset_orphans_tbl.hpp
)
set(OP_VERSION_TBL_PATH ${CMAKE_CURRENT_BINARY_DIR}/op_version_tbl.hpp)
add_custom_command(
    OUTPUT ${OP_VERSION_TBL_PATH}
    DEPENDS ${OPSETS}
    COMMAND ${CMAKE_COMMAND}
        -D OPSETS="${OPSETS}"
        -D CURRENT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -Wno-dev
        -P "${PROJECT_SOURCE_DIR}/cmake/Modules/gen_op_version.cmake"
)
add_custom_target(
    gen_op_version_table
    ALL
    DEPENDS ${OP_VERSION_TBL_PATH}
)

if(NGRAPH_JSON_ENABLE)
    list(APPEND SRC serializer.cpp serializer.hpp)
else()
    list(APPEND SRC serializer_stub.cpp)
endif()

configure_file(version.in.hpp version.hpp)

if (NGRAPH_STATIC_LIB_ENABLE)
    add_library(ngraph STATIC ${SRC})
else()
    add_library(ngraph SHARED ${SRC})
endif()

set_target_properties(ngraph PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      C_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)

if(NGRAPH_STL_DEBUG_ENABLE)
    target_compile_definitions(ngraph PUBLIC ${NGRAPH_STL_DEBUG_DEFINITION})
endif()

if(NOT NGRAPH_JSON_ENABLE)
    target_compile_definitions(ngraph PUBLIC NGRAPH_JSON_DISABLE)
endif()

if(NGRAPH_INTERPRETER_STATIC_LIB_ENABLE)
    target_compile_definitions(ngraph PUBLIC NGRAPH_INTERPRETER_STATIC_LIB_ENABLE)
endif()

if(NGRAPH_CPU_STATIC_LIB_ENABLE)
    target_compile_definitions(ngraph PUBLIC NGRAPH_CPU_STATIC_LIB_ENABLE)
endif()

if(NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
    set_property(SOURCE runtime/backend.cpp APPEND PROPERTY COMPILE_DEFINITIONS NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
    set_property(SOURCE runtime/backend_manager.cpp APPEND PROPERTY COMPILE_DEFINITIONS NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
endif()

set_property(SOURCE runtime/backend_manager.cpp APPEND PROPERTY COMPILE_DEFINITIONS SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}")
set_property(SOURCE runtime/backend_manager.cpp APPEND PROPERTY COMPILE_DEFINITIONS SHARED_LIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_subdirectory(frontend)

find_package(Graphviz QUIET)
if (GRAPHVIZ_FOUND)
    set_property(SOURCE pass/visualize_tree.cpp APPEND PROPERTY COMPILE_DEFINITIONS GRAPHVIZ_FOUND)
endif()

if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
endif()

add_subdirectory(codegen)
add_subdirectory(runtime)

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(ngraph PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()
if(NGRAPH_JSON_ENABLE)
    target_link_libraries(ngraph PRIVATE nlohmann_json::nlohmann_json)
endif()
target_compile_definitions(ngraph PUBLIC NGRAPH_VERSION="${NGRAPH_VERSION}")
add_dependencies(ngraph gen_op_version_table)

if (LINUX)
    # nGraph links against one or more libraries (ex. LLVM) but we don't want to
    # export these symbols as part of the DSO. This is a GNU ld (and derivatives) specific
    # option so making this portable is still an open issue. As a note for the future,
    # this is not an issue on Windows and LLVM's lld does support --exclude-libs.
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--exclude-libs,ALL")

    # GCC invokes the linker with --as-needed by default which doesn't work for us
    # because generated code needs to find symbols in these DSOs at runtime.
    # The fix below is temporary and will be removed once we find a better way
    # to do this because certain dependencies like the OpenMP runtime libraries
    # _do_ need to be linked with --as-needed with a higher priority for the
    # Intel OpenMP runtime so we don't mix libgomp and libiomp5
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--no-as-needed")
    endif()
elseif(APPLE)
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-rpath,@loader_path")
endif()

# Defines macro in C++ to load backend plugin
target_include_directories(ngraph PUBLIC $<BUILD_INTERFACE:${NGRAPH_INCLUDE_PATH}> $<INSTALL_INTERFACE:include>)
target_include_directories(ngraph PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>)

#Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::ngraph ALIAS ngraph)

if (NOT WIN32)
    target_link_libraries(ngraph PRIVATE dl)
endif()

# Build subdirectories for all build types on Windows
if(WIN32)
    foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
        if(NOT EXISTS ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
            file(MAKE_DIRECTORY ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
        endif()
    endforeach()
endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

# nGraph
install(TARGETS ngraph EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph
    COMPONENT ngraph
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.hpp
    DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph
    COMPONENT ngraph)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A most excellent graph library")
set(CPACK_PACKAGE_NAME "nGraph")
set(CPACK_PACKAGE_CONTACT "Robert Kimball")
# set(CPACK_PACKAGE_VERSION
set(CPACK_PACKAGE_VENDOR "Intel Nervana")
# set(CPACK_PACKAGE_DESCRIPTION_FILE "")

set(CPACK_PACKAGE_VERSION_MAJOR ${NGRAPH_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${NGRAPH_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${NGRAPH_VERSION_PATCH})
include(CPack)
