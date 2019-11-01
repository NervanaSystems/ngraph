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

include(FindOpenMP)

if (NGRAPH_CPU_STATIC_LIB_ENABLE)
    set(LIBRARY_TYPE STATIC)
else()
    set(LIBRARY_TYPE SHARED)
endif()

set(SRC
    cpu_backend.cpp
    cpu_builder.cpp
    cpu_builder_registry.cpp
    cpu_call_frame.cpp
    cpu_executor.cpp
    cpu_external_function.cpp
    cpu_kernels.cpp
    cpu_layout_descriptor.cpp
    cpu_op_annotations.cpp
    cpu_tensor_view_wrapper.cpp
    cpu_tensor_view.cpp
    cpu_tracing.cpp
    cpu_visualize_tree.cpp
    cpu_cse.cpp
    cpu_debugger.cpp
    cpu_debug_tracer.cpp
    builder/add.cpp
    builder/allreduce.cpp
    builder/avg_pool.cpp
    builder/argmin.cpp
    builder/argmax.cpp
    builder/batch_norm.cpp
    builder/broadcast.cpp
    builder/broadcast_distributed.cpp
    builder/bounded_relu.cpp
    builder/concat.cpp
    builder/convert.cpp
    builder/convert_layout.cpp
    builder/convolution.cpp
    builder/dot.cpp
    builder/dropout.cpp
    builder/embedding_lookup.cpp
    builder/erf.cpp
    builder/gather.cpp
    builder/gather_nd.cpp
    builder/gelu.cpp
    builder/leaky_relu.cpp
    builder/lstm.cpp
    builder/lrn.cpp
    builder/matmul_bias.cpp
    builder/max.cpp
    builder/max_pool.cpp
    builder/min.cpp
    builder/one_hot.cpp
    builder/random_uniform.cpp
    builder/relu.cpp
    builder/pad.cpp
    builder/product.cpp
    builder/reduce_function.cpp
    builder/replace_slice.cpp
    builder/quantization.cpp
    builder/quantized_conv.cpp
    builder/quantized_dot.cpp
    builder/quantized_matmul.cpp
    builder/reshape.cpp
    builder/reverse.cpp
    builder/reverse_sequence.cpp
    builder/rnn.cpp
    builder/scatter_add.cpp
    builder/scatter_nd_add.cpp
    builder/select.cpp
    builder/sigmoid.cpp
    builder/slice.cpp
    builder/state.cpp
    builder/softmax.cpp
    builder/get_output_element.cpp
    builder/sum.cpp
    builder/tile.cpp
    builder/topk.cpp
    builder/update_slice.cpp
    kernel/pad.cpp
    kernel/reduce_max.cpp
    kernel/reduce_sum.cpp
    kernel/reshape.cpp
    mkldnn_emitter.cpp
    mkldnn_invoke.cpp
    mkldnn_utils.cpp
    op/batch_mat_mul_transpose.cpp
    op/batch_norm_relu.cpp
    op/bounded_relu.cpp
    op/conv_add.cpp
    op/conv_relu.cpp
    op/convert_layout.cpp
    op/deconv.cpp
    op/dropout.cpp
    op/gelu_backprop.cpp
    op/group_conv_bias.cpp
    op/halide_op.cpp
    op/leaky_relu.cpp
    op/lstm.cpp
    op/matmul_bias.cpp
    op/max_pool_with_indices.cpp
    op/quantized_matmul.cpp
    op/rnn.cpp
    op/sigmoid_mul.cpp
    op/update_slice.cpp
    pass/cpu_assignment.cpp
    pass/cpu_collapse_dims.cpp
    pass/cpu_fusion.cpp
    pass/cpu_horizontal_fusion.cpp
    pass/cpu_layout.cpp
    pass/cpu_mat_fusion.cpp
    pass/cpu_memory_assignment.cpp
    pass/cpu_memory_optimization.cpp
    pass/cpu_post_layout_optimizations.cpp
    pass/cpu_rnn_fusion.cpp
    pass/cpu_workspace_insertion.cpp
)

if (NOT NGRAPH_DEX_ONLY)
    set(SRC
        ${SRC}
        cpu_emitter.cpp
        cpu_kernel_emitters.cpp
        cpu_kernel_utils.cpp
        pass/cpu_mkldnn_primitive_build.cpp
        )
endif()

if (NGRAPH_HALIDE)
    set(SRC
        ${SRC}
        builder/compiled_kernel.cpp
        builder/halide_op.cpp
        builder/halide_generators.cpp
        pass/halide_subgraph_extraction.cpp
        )
endif()

if (NGRAPH_MLIR_ENABLE)
    set(SRC
        ${SRC}
        builder/mlir_cpu_compiled_kernel.cpp
        )
endif()

set(NGRAPH_CPU_ALL_DATATYPES
    boolean
    f32
    f64
    i8
    i16
    i32
    i64
    u8
    u16
    u32
    u64
    )

set(NGRAPH_CPU_COMMON_DATATYPES
    f32
    i64
    )


if (NGRAPH_CPU_ENABLE)
    set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")

    set(NGRAPH_CPU_OPTIMIZED_DATATYPES "common"
        CACHE STRING "Semicolon-separated list of datatypes to optimize for, or \"common\" or \"all\".")

    if (NGRAPH_CPU_OPTIMIZED_DATATYPES STREQUAL "all")
        set(NGRAPH_CPU_OPTIMIZED_DATATYPES ${NGRAPH_CPU_ALL_DATATYPES})
    endif()

    if (NGRAPH_CPU_OPTIMIZED_DATATYPES STREQUAL "common")
        set(NGRAPH_CPU_OPTIMIZED_DATATYPES ${NGRAPH_CPU_COMMON_DATATYPES})
    endif()

    list(REMOVE_DUPLICATES NGRAPH_CPU_OPTIMIZED_DATATYPES)

    add_library(cpu_backend ${LIBRARY_TYPE} ${SRC})
    if (NGRAPH_CPU_STATIC_LIB_ENABLE)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_CPU_STATIC_LIB_ENABLE")
    endif()
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(cpu_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    if (NGRAPH_USE_LEGACY_MKLDNN)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_USE_LEGACY_MKLDNN")
    endif()
    if (NGRAPH_DEX_ONLY)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_DEX_ONLY")
    endif()
    if (NGRAPH_ENABLE_CPU_CONV_AUTO)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_ENABLE_CPU_CONV_AUTO")
    endif()
    if (NGRAPH_TBB_ENABLE)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_TBB_ENABLE")
    endif()
    if (NGRAPH_HALIDE)
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_HALIDE")
        ExternalProject_Get_Property(ext_halide BINARY_DIR)
        target_include_directories(cpu_backend SYSTEM PRIVATE ${BINARY_DIR}/include)
        target_link_libraries(cpu_backend PRIVATE ${BINARY_DIR}/lib/libHalide.a libhalidellvm)
        add_dependencies(cpu_backend ext_halide)
    endif()

    if(NOT NGRAPH_FAST_MATH_ENABLE)
        target_compile_definitions(cpu_backend PRIVATE EIGEN_FAST_MATH=0)
    endif()

    if(OPENMP_FOUND)
        target_compile_options(cpu_backend PRIVATE "${OpenMP_CXX_FLAGS}")
        if (NOT WIN32)
            target_compile_definitions(cpu_backend PRIVATE EIGEN_OPENMP)
        endif()
    else()
         message(WARNING "The build toolset doesn't support OpenMP. This will impact performance and lead to slowdowns.")
    endif()

    if (MSVS)
        target_compile_definitions(cpu_backend PRIVATE EIGEN_HAS_CONSTEXPR)
        # under debug mode, more files besides builder/dot.cpp rises the error
        target_compile_options(cpu_backend PRIVATE "/bigobj" )
    endif()
    target_compile_definitions(cpu_backend PRIVATE CPU_BACKEND_DLL_EXPORTS)

    foreach(t ${NGRAPH_CPU_OPTIMIZED_DATATYPES})
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_CPU_OPTIMIZE_${t}")
    endforeach()

    add_dependencies(cpu_backend libmkldnn ext_eigen)
   	target_link_libraries(cpu_backend PUBLIC ngraph libmkldnn libmkl libeigen)
    if (NGRAPH_JSON_ENABLE)
        target_link_libraries(cpu_backend PUBLIC libjson)
    endif()    
    if (NGRAPH_TBB_ENABLE)
        target_link_libraries(cpu_backend PUBLIC libtbb)
    endif()
    if (NOT NGRAPH_DEX_ONLY)
        target_link_libraries(cpu_backend PRIVATE codegen)
    endif()
    target_include_directories(cpu_backend SYSTEM PUBLIC libmkldnn)

    if (NOT APPLE AND NOT MSVS)
        # CPU backend uses third-party libraries like Eigen that might be linked in and
        # exported by other DSOs as well. In the absence of versioning, this could lead to the
        # CPU backend picking up the wrong version or even multiple versions of the
        # third-party library. -Bsymbolic-functions tells the linker to prefer the internal
        # version inside cpu_backend over what is available through the global symbol table
        set_property(TARGET cpu_backend APPEND PROPERTY LINK_FLAGS "-Wl,-Bsymbolic-functions -Wl,--exclude-libs=ALL")
    endif()

    if (NGRAPH_MLIR_ENABLE)
        # TODO: can we get away without LLVM/MLIR include path. 
        # Currently mlir backend compiler.hpp include LLVM/MLIR files
        get_directory_property(MLIR_LLVM_INCLUDEPATH
                               DIRECTORY ${NGRAPH_MLIR_SOURCE_DIR}
                               DEFINITION MLIR_LLVM_INCLUDEPATH)
        
        message(STATUS "Building CPU backend with MLIR")
        add_dependencies(cpu_backend mlir_backend)
        target_include_directories(cpu_backend PUBLIC ${MLIR_INCLUDE_PATHS} ${MLIR_LLVM_INCLUDE_PATH})
        target_link_libraries(cpu_backend PUBLIC mlir_backend)
        # TODO: Get rid of the compile time def, and move all MLIR code to separate src files
        # and add them to cpu_backend here instead.
        target_compile_definitions(cpu_backend PRIVATE "NGRAPH_MLIR_ENABLE")
    endif()
    
    install(TARGETS cpu_backend DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
