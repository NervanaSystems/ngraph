# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set (SRC
    autodiff/adjoints.cpp
    autodiff/backprop_derivative.cpp
    autodiff/backprop_function.cpp
    autodiff/numeric_derivative.cpp
    descriptor/input.cpp
    descriptor/layout/dense_tensor_view_layout.cpp
    descriptor/layout/tensor_view_layout.cpp
    descriptor/output.cpp
    descriptor/primary_tensor_view.cpp
    descriptor/tensor.cpp
    descriptor/tensor_view.cpp
    descriptor/tuple.cpp
    function.cpp
    log.cpp
    node.cpp
    ops/abs.cpp
    ops/add.cpp
    ops/binary_elementwise_arithmetic.cpp
    ops/binary_elementwise_builtin.cpp
    ops/binary_elementwise_comparison.cpp
    ops/broadcast.cpp
    ops/concatenate.cpp
    ops/constant.cpp
    ops/convert.cpp
    ops/cos.cpp
    ops/divide.cpp
    ops/dot.cpp
    ops/exp.cpp
    ops/function_call.cpp
    ops/get_tuple_element.cpp
    ops/log.cpp
    ops/maximum.cpp
    ops/minimum.cpp
    ops/multiply.cpp
    ops/negative.cpp
    ops/not.cpp
    ops/op.cpp
    ops/parameter.cpp
    ops/reduce.cpp
    ops/reshape.cpp
    ops/select.cpp
    ops/sin.cpp
    ops/slice.cpp
    ops/subtract.cpp
    ops/sum.cpp
    ops/tan.cpp
    ops/tuple.cpp
    ops/unary_elementwise_arithmetic.cpp
    ops/unary_elementwise_builtin.cpp
    pass/assign_tensors.cpp
    pass/collect_functions.cpp
    pass/dump_sorted.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/pass.cpp
    pass/propagate_types.cpp
    pass/topological_sort.cpp
    pass/visualize_tree.cpp
    runtime/backend.cpp
    runtime/manager.cpp
    runtime/ngvm/call_frame.cpp
    runtime/ngvm/external_function.cpp
    runtime/ngvm/ngvm_backend.cpp
    runtime/ngvm/ngvm_manager.cpp
    runtime/tensor_view.cpp
    runtime/tuple.cpp
    runtime/utils.cpp
    shape.cpp
    types/element_type.cpp
    types/type.cpp
    util.cpp
    )

# find_program (GRAPHVIZ dot)
# message (STATUS "graphviz '${GRAPHVIZ}'")
find_package(Graphviz)
if (GRAPHVIZ_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGRAPHVIZ_FOUND")
endif()

include_directories(
    "${NGRAPH_INCLUDE_PATH}"
    )

include_directories(
    SYSTEM
    "${EIGEN_INCLUDE_DIR}"
    )

add_library(ngraph SHARED ${SRC})
target_include_directories(ngraph PUBLIC "${NGRAPH_INCLUDE_PATH}")

if (APPLE)
    set_property(TARGET ngraph PROPERTY PREFIX "lib")
    set_property(TARGET ngraph PROPERTY OUTPUT_NAME "ngraph.so")
    set_property(TARGET ngraph PROPERTY SUFFIX "")
else()
    include_directories("${MKLDNN_INCLUDE_DIR}")
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

# External
# Requirements:
# - In NGraph, there are multiple include paths set for different external projects.
# - However, we want one single include path for external projects from the TF bridge side, while
#   making sure that the `#include` files can still be resolved.
# - Therefore, when `make install`, the include directory will be "flattened" for each external
#   projects respectively.

install(DIRECTORY
    ${EIGEN_INCLUDE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}"
)
if (NOT APPLE)
install(DIRECTORY
    ${MKLDNN_INCLUDE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}"
)
install(DIRECTORY
    ${MKLDNN_LIB_DIR}/
    DESTINATION "${NGRAPH_INSTALL_LIB}"
)
endif()

add_dependencies(ngraph eigen)
