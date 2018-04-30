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
    op/abs.cpp
    op/acos.cpp
    op/add.cpp
    op/allreduce.cpp
    op/and.cpp
    op/asin.cpp
    op/atan.cpp
    op/avg_pool.cpp
    op/batch_norm.cpp
    op/broadcast.cpp
    op/ceiling.cpp
    op/concat.cpp
    op/constant.cpp
    op/convert.cpp
    op/convolution.cpp
    op/cos.cpp
    op/cosh.cpp
    op/divide.cpp
    op/dot.cpp
    op/equal.cpp
    op/exp.cpp
    op/floor.cpp
    op/function_call.cpp
    op/get_output_element.cpp
    op/greater.cpp
    op/greater_eq.cpp
    op/less.cpp
    op/less_eq.cpp
    op/log.cpp
    op/max.cpp
    op/maximum.cpp
    op/max_pool.cpp
    op/min.cpp
    op/minimum.cpp
    op/multiply.cpp
    op/negative.cpp
    op/not.cpp
    op/not_equal.cpp
    op/one_hot.cpp
    op/op.cpp
    op/or.cpp
    op/pad.cpp
    op/parameter.cpp
    op/power.cpp
    op/product.cpp
    op/reduce.cpp
    op/reduce_window.cpp
    op/relu.cpp
    op/remainder.cpp
    op/replace_slice.cpp
    op/reshape.cpp
    op/result.cpp
    op/reverse.cpp
    op/reverse_sequence.cpp
    op/select_and_scatter.cpp
    op/select.cpp
    op/sign.cpp
    op/sin.cpp
    op/sinh.cpp
    op/slice.cpp
    op/softmax.cpp
    op/sqrt.cpp
    op/subtract.cpp
    op/sum.cpp
    op/tan.cpp
    op/tanh.cpp
    op/util/arithmetic_reduction.cpp
    op/util/binary_elementwise_arithmetic.cpp
    op/util/binary_elementwise_comparison.cpp
    op/util/binary_elementwise_logical.cpp
    op/util/binary_elementwise.cpp
    op/util/requires_tensor_view_args.cpp
    op/util/unary_elementwise_arithmetic.cpp
    op/util/unary_elementwise.cpp
    pass/assign_placement.cpp
    pass/algebraic_simplification.cpp
    pass/cse.cpp
    pass/dump_sorted.cpp
    pass/get_output_element_elimination.cpp
    pass/graph_rewrite.cpp
    pass/inliner.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/nop_elimination.cpp
    pass/pass.cpp
    pass/reshape_elimination.cpp
    pass/result_copy_elimination.cpp
    pass/zero_dim_tensor_elimination.cpp
    pass/validate_graph.cpp
    pass/visualize_tree.cpp
    pass/core_fusion.cpp
    pass/zero_dim_tensor_elimination.cpp
    pattern/matcher.cpp
    runtime/aligned_buffer.cpp
    runtime/backend.cpp
    runtime/host_tensor_view.cpp
    runtime/tensor_view.cpp
    serializer.cpp
    shape.cpp
    type/element_type.cpp
    type/type.cpp
    util.cpp
    graph_util.cpp
    placement.cpp
    cpio.cpp
    )

find_package(Graphviz QUIET)
if (GRAPHVIZ_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGRAPHVIZ_FOUND")
endif()

if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

include_directories("${NGRAPH_INCLUDE_PATH}")

add_subdirectory(codegen)
add_subdirectory(runtime)

add_library(ngraph SHARED ${SRC})
add_dependencies(ngraph ext_json)
target_link_libraries(ngraph PUBLIC libjson)

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
target_include_directories(ngraph PUBLIC "${NGRAPH_INCLUDE_PATH}")
target_link_libraries(ngraph PUBLIC dl pthread)

if (APPLE)
    set_property(TARGET ngraph PROPERTY PREFIX "lib")
    set_property(TARGET ngraph PROPERTY OUTPUT_NAME "ngraph.so")
    set_property(TARGET ngraph PROPERTY SUFFIX "")
endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

# NGraph
install(FILES ${CMAKE_BINARY_DIR}/VERSION DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS ngraph DESTINATION ${NGRAPH_INSTALL_LIB})  # libngraph.so
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph"
    FILES_MATCHING PATTERN "*.hpp"
)

if(MKLDNN_LIB_DIR)
    install(DIRECTORY
        ${MKLDNN_LIB_DIR}/
        DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()

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
