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

set (SRC
    autodiff/adjoints.cpp
    builder/autobroadcast.cpp
    builder/numpy_transpose.cpp
    builder/reduce_ops.cpp
    builder/xla_tuple.cpp
    coordinate_transform.cpp
    descriptor/input.cpp
    descriptor/layout/dense_tensor_view_layout.cpp
    descriptor/layout/tensor_view_layout.cpp
    descriptor/output.cpp
    descriptor/primary_tensor_view.cpp
    descriptor/tensor.cpp
    descriptor/tensor_view.cpp
    file_util.cpp
    function.cpp
    log.cpp
    node.cpp
    ops/abs.cpp
    ops/add.cpp
    ops/allreduce.cpp
    ops/avg_pool.cpp
    ops/batch_norm.cpp
    ops/broadcast.cpp
    ops/concat.cpp
    ops/constant.cpp
    ops/convert.cpp
    ops/convolution.cpp
    ops/cos.cpp
    ops/cosh.cpp
    ops/divide.cpp
    ops/dot.cpp
    ops/exp.cpp
    ops/function_call.cpp
    ops/get_output_element.cpp
    ops/log.cpp
    ops/max_pool.cpp
    ops/maximum.cpp
    ops/minimum.cpp
    ops/multiply.cpp
    ops/negative.cpp
    ops/not.cpp
    ops/one_hot.cpp
    ops/op.cpp
    ops/pad.cpp
    ops/parameter.cpp
    ops/power.cpp
    ops/reduce.cpp
    ops/reduce_window.cpp
    ops/relu.cpp
    ops/replace_slice.cpp
    ops/reshape.cpp
    ops/reverse.cpp
    ops/result.cpp
    ops/select.cpp
    ops/select_and_scatter.cpp
    ops/sin.cpp
    ops/sinh.cpp
    ops/slice.cpp
    ops/softmax.cpp
    ops/sqrt.cpp
    ops/subtract.cpp
    ops/sum.cpp
    ops/tan.cpp
    ops/tanh.cpp
    ops/util/arithmetic_reduction.cpp
    ops/util/binary_elementwise_arithmetic.cpp
    ops/util/binary_elementwise_comparison.cpp
    ops/util/binary_elementwise.cpp
    ops/util/requires_tensor_view_args.cpp
    ops/util/unary_elementwise_arithmetic.cpp
    ops/util/unary_elementwise.cpp
    pass/assign_placement.cpp
    pass/dump_sorted.cpp
    pass/graph_rewrite.cpp
    pass/inliner.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/pass.cpp
    pass/reshape_elimination.cpp
    pass/result_copy_elimination.cpp
    pass/visualize_tree.cpp
    pattern/matcher.cpp
    pattern/core_fusion.cpp
    runtime/aligned_buffer.cpp
    runtime/host_tensor_view.cpp
    runtime/interpreter/int_backend.cpp
    runtime/interpreter/int_call_frame.cpp
    runtime/interpreter/int_external_function.cpp
    runtime/interpreter/int_kernels.cpp
    runtime/interpreter/int_manager.cpp
    runtime/manager.cpp
    runtime/tensor_view.cpp
    serializer.cpp
    shape.cpp
    types/element_type.cpp
    types/type.cpp
    util.cpp
    graph_util.cpp
    placement.cpp
    )

message(STATUS  ${CMAKE_CURRENT_SOURCE_DIR}/ops)
file(GLOB_RECURSE OPS "${CMAKE_CURRENT_SOURCE_DIR}/ops" "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.hpp")
foreach(OP ${OPS})
    file(STRINGS ${OP} OP_CLASS REGEX "class [A-Za-z0-9_]+ :")
    foreach(LINE ${OP_CLASS})
        string(REGEX REPLACE ".*class ([A-Za-z0-9_]+) : public ([A-Za-z0-9_]+).*" "\\1:\\2" CLASS_FOUND ${LINE})
        set(OP_CLASS_LIST ${OP_CLASS_LIST} ${CLASS_FOUND})
    endforeach(LINE ${OP_CLASS})
endforeach()
message(STATUS "${CMAKE_CURRENT_BINARY_DIR}/ops_list.txt")
string(REPLACE ";" "\n" OP_CLASS_LINES "${OP_CLASS_LIST}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ops_list.txt" "${OP_CLASS_LINES}")

# find_program (GRAPHVIZ dot)
# message (STATUS "graphviz '${GRAPHVIZ}'")
find_package(Graphviz QUIET)
if (GRAPHVIZ_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGRAPHVIZ_FOUND")
endif()
if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

include_directories("${NGRAPH_INCLUDE_PATH}")
include_directories(SYSTEM "${EIGEN_INCLUDE_DIR}")

if (NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR AND
    MKLDNN_INCLUDE_DIR)
    find_package(ZLIB REQUIRED)

    if (NGRAPH_TBB_ENABLE)
        include(${TBB_ROOT}/cmake/TBBBuild.cmake)
        tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/tbb_build
                  tbb_build_prefix=tbb CONFIG_DIR TBB_DIR)
        find_package(TBB REQUIRED tbb)
        if (NOT TBB_FOUND)
            message(FATAL_ERROR "TBB is needed by the CPU backend and was not found")
        else()
            message(STATUS "Found TBB and imported target ${TBB_IMPORTED_TARGETS}")
        endif()
    endif()

    include_directories(SYSTEM ${LLVM_INCLUDE_DIR} ${MKLDNN_INCLUDE_DIR})
    link_directories(${LLVM_LIB_DIR} ${MKLDNN_LIB_DIR})

    # Add sources for the CPU backend
    # and all its dependencies
    set(SRC ${SRC}
        codegen/code_writer.cpp
        codegen/compiler.cpp
        codegen/execution_engine.cpp
        runtime/cpu/cpu_call_frame.cpp
        runtime/cpu/cpu_backend.cpp
        runtime/cpu/cpu_manager.cpp
        runtime/cpu/cpu_kernels.cpp
        runtime/cpu/cpu_kernel_emitters.cpp
        runtime/cpu/cpu_kernel_utils.cpp
        runtime/cpu/cpu_emitter.cpp
        runtime/cpu/cpu_external_function.cpp
        runtime/cpu/cpu_tensor_view.cpp
        runtime/cpu/cpu_tensor_view_wrapper.cpp
        runtime/cpu/cpu_layout_descriptor.cpp
        runtime/cpu/cpu_tracing.cpp
        runtime/cpu/mkldnn_emitter.cpp
        runtime/cpu/mkldnn_invoke.cpp
        runtime/cpu/mkldnn_utils.cpp
        runtime/cpu/ops/conv_bias.cpp
        runtime/cpu/ops/convert_layout.cpp
        runtime/cpu/ops/sigmoid.cpp
        runtime/cpu/ops/matmul_bias.cpp
        runtime/cpu/pass/cpu_assignment.cpp
        runtime/cpu/pass/cpu_fusion.cpp
        runtime/cpu/pass/cpu_layout.cpp
        runtime/cpu/pass/cpu_nop_elimination.cpp
    )
    # LLVM binary builds are typically built without RTTI
    # The built-in headers are in a version-specific directory
    # This must be kept in sync with the LLVM + Clang version in use
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

    set(HEADER_SEARCH_DEFINES
        "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\""
        "MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\""
        "CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.1/include\""
        "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
        "INSTALLED_HEADERS_PATH=\"${CMAKE_INSTALL_PREFIX}/include\""
    )

    if (NGRAPH_TBB_ENABLE)
        set_source_files_properties(runtime/cpu/cpu_external_function.cpp PROPERTIES COMPILE_DEFINITIONS "NGRAPH_TBB_ENABLE")
        set(HEADER_SEARCH_DEFINES ${HEADER_SEARCH_DEFINES}
            "TBB_HEADERS_PATH=\"${TBB_ROOT}/include\""
            "NGRAPH_TBB_ENABLE"
        )
    endif()
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES};")
    set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")

if(NGRAPH_DISTRIBUTED_ENABLE AND MPI_CXX_INCLUDE_PATH)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

    set_property(SOURCE codegen/compiler.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "MPI_HEADER_PATH=\"${MPI_C_INCLUDE_PATH}\";")
endif()

    # GPU backend current requires CPU because they share compiler.cpp,
    # and compiler.cpp requires MKLDNN
    if(NGRAPH_GPU_ENABLE)
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})

        # Add sources for the GPU backend
        # and all its dependencies
        set(SRC ${SRC}
            runtime/gpu/gpu_call_frame.cpp
            runtime/gpu/gpu_backend.cpp
            runtime/gpu/gpu_emitter.cpp
            runtime/gpu/gpu_external_function.cpp
            runtime/gpu/gpu_kernel_emitters.cpp
            runtime/gpu/gpu_manager.cpp
            runtime/gpu/gpu_tensor_view.cpp
            runtime/gpu/gpu_tensor_view_wrapper.cpp
            runtime/gpu/gpu_util.cpp
            runtime/gpu/gpu_cuda_kernel_emitters.cpp
            runtime/gpu/gpu_cuda_kernel_builder.cpp
            runtime/gpu/gpu_cuda_function_builder.cpp
            runtime/gpu/gpu_cuda_function_pool.cpp
            runtime/gpu/gpu_cuda_context_manager.cpp
            )
        set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
        "CUDA_HEADER_PATHS=\"${CUDA_INCLUDE_DIRS}\";")
    endif()

endif()

if (NGRAPH_ARGON_ENABLE)
    link_directories(${ARGON_TRANSFORMER_LIB_DIR})
    include_directories(${ARGON_TRANSFORMER_INCLUDE_DIR})
endif()

add_library(ngraph SHARED ${SRC})
if (NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR AND MKLDNN_INCLUDE_DIR)
    # Generate the resource file containing all headers used by the codegen compiler
    add_custom_target(header_resource
        resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
        DEPENDS resource_generator ext_eigen ext_llvm ext_mkldnn
        BYPRODUCTS
    )
    add_dependencies(ngraph header_resource ext_json)
    include_directories(SYSTEM ${CMAKE_BINARY_DIR})
    include_directories(SYSTEM ${JSON_INCLUDE_DIR})
endif()

if (NOT APPLE)
    # NGraph links against one or more libraries (ex. LLVM) but we don't want to
    # export these symbols as part of the DSO. This is a GNU ld (and derivatives) specific
    # option so making this portable is still an open issue. As a note for the future,
    # this is not an issue on Windows and LLVM's lld does support --exclude-libs.
    set_target_properties(ngraph PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

    # GCC invokes the linker with --as-needed by default which doesn't work for us
    # because generated code needs to find symbols in these DSOs at runtime.
    # The fix below is temporary and will be removed once we find a better way
    # to do this because certain dependencies like the OpenMP runtime libraries
    # _do_ need to be linked with --as-needed with a higher priority for the
    # Intel OpenMP runtime so we don't mix libgomp and libiomp5
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--no-as-needed")
    endif()
endif()

# Defines macro in C++ to load backend plugin
if (NGRAPH_ARGON_ENABLE)
    target_compile_definitions(ngraph PRIVATE NGRAPH_ARGON_ENABLE)
endif()

target_include_directories(ngraph PUBLIC "${NGRAPH_INCLUDE_PATH}")

if((NGRAPH_CPU_ENABLE OR NGRAPH_GPU_ENABLE) AND LLVM_LINK_LIBS)
    target_link_libraries(ngraph PRIVATE ${LLVM_LINK_LIBS})
endif()

if (APPLE)
    set_property(TARGET ngraph PROPERTY PREFIX "lib")
    set_property(TARGET ngraph PROPERTY OUTPUT_NAME "ngraph.so")
    set_property(TARGET ngraph PROPERTY SUFFIX "")
endif()

if(NGRAPH_CPU_ENABLE AND MKLDNN_LIB_DIR)
    target_link_libraries(ngraph PRIVATE mkldnn)
endif()

if(NGRAPH_CPU_ENABLE)
    target_link_libraries(ngraph PRIVATE ${TBB_IMPORTED_TARGETS})
endif()

# Nvidia
if(NGRAPH_GPU_ENABLE AND CUDA_LIBRARIES)
    find_library(CUDA_nvrtc_LIBRARY nvrtc /usr/local/cuda/lib64)
    find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
    target_link_libraries(ngraph PUBLIC ${CUDA_cuda_LIBRARY} ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES})
endif()

if(MPI_CXX_INCLUDE_PATH)
    target_link_libraries(ngraph PRIVATE ${MPI_CXX_LIBRARIES})
endif()

# Argon
if (NGRAPH_ARGON_ENABLE)
    target_link_libraries(ngraph PRIVATE ${ARGON_TRANSFORMER_LIB_DIR}/libargon.so)
    target_link_libraries(ngraph PRIVATE ${ARGON_API_LIBS})
    add_dependencies(ngraph ext_argon_transformer)
    add_dependencies(ext_argon_transformer ext_llvm)
endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

# NGraph
install(TARGETS ngraph DESTINATION ${NGRAPH_INSTALL_LIB})  # libngraph.so
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph"
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY
    ${MKLDNN_LIB_DIR}/
    DESTINATION "${NGRAPH_INSTALL_LIB}"
)

if (NGRAPH_TBB_ENABLE)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/tbb_release/
        DESTINATION ${NGRAPH_INSTALL_LIB}
        FILES_MATCHING REGEX "/libtbb${CMAKE_SHARED_LIBRARY_SUFFIX}(\\.[0-9]+)*$"
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/tbb_debug/
        DESTINATION ${NGRAPH_INSTALL_LIB}
        FILES_MATCHING REGEX "/libtbb_debug${CMAKE_SHARED_LIBRARY_SUFFIX}(\\.[0-9]+)*$"
    )
endif()

if (NGRAPH_ARGON_ENABLE)
    install(DIRECTORY ${ARGON_TRANSFORMER_LIB_DIR}/
        DESTINATION ${NGRAPH_INSTALL_LIB}
        FILES_MATCHING PATTERN "*.so"
    )
    install(DIRECTORY ${ARGON_TRANSFORMER_INCLUDE_DIR}/
        DESTINATION ${NGRAPH_INSTALL_INCLUDE}
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(DIRECTORY ${ARGON_TRANSFORMER_INCLUDE_DIR}/
        DESTINATION ${NGRAPH_INSTALL_INCLUDE}
        FILES_MATCHING PATTERN "*.h"
    )
endif()
