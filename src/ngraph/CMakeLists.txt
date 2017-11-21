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
    descriptor/input.cpp
    descriptor/layout/dense_tensor_view_layout.cpp
    descriptor/layout/tensor_view_layout.cpp
    descriptor/output.cpp
    descriptor/primary_tensor_view.cpp
    descriptor/tensor.cpp
    descriptor/tensor_view.cpp
    descriptor/tuple.cpp
    file_util.cpp
    function.cpp
    log.cpp
    node.cpp
    ops/add.cpp
    ops/binary_elementwise_arithmetic.cpp
    ops/binary_elementwise_comparison.cpp
    ops/binary_elementwise.cpp
    ops/broadcast.cpp
    ops/concatenate.cpp
    ops/constant.cpp
    ops/convert.cpp
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
    ops/op.cpp
    ops/parameter.cpp
    ops/reduce.cpp
    ops/reshape.cpp
    ops/select.cpp
    ops/slice.cpp
    ops/subtract.cpp
    ops/sum.cpp
    ops/tuple.cpp
    ops/unary_elementwise_arithmetic.cpp
    ops/unary_elementwise.cpp
    pass/dump_sorted.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/pass.cpp
    pass/topological_sort.cpp
    pass/visualize_tree.cpp
    runtime/aligned_buffer.cpp
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

if (LLVM_INCLUDE_DIR AND
    MKLDNN_INCLUDE_DIR)
    find_package(ZLIB REQUIRED)

    include_directories(SYSTEM ${LLVM_INCLUDE_DIR} ${MKLDNN_INCLUDE_DIR})
    link_directories(${LLVM_LIB_DIR} ${MKLDNN_LIB_DIR})

    # Add sources for the CPU backend
    # and all its dependencies
    set(SRC ${SRC}
        codegen/code_writer.cpp
        codegen/compiler.cpp
        codegen/execution_engine.cpp
        runtime/cpu/call_frame.cpp
        runtime/cpu/cpu_backend.cpp
        runtime/cpu/cpu_manager.cpp
        runtime/cpu/cpu_kernels.cpp
        runtime/cpu/emitter.cpp
        runtime/cpu/external_function.cpp
        runtime/cpu/tensor_view.cpp
        )
    # LLVM binary builds are typically built without RTTI
    # The built-in headers are in a version-specific directory
    # This must be kept in sync with the LLVM + Clang version in use
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_DEFINITIONS
      "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";")
    set(NGRAPH_CPU_PCH_ENABLE 0 CACHE STRING "Enable pre-compiled headers in the CPU backend")
    set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")
    set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
      "NGCPU_PCH=${NGRAPH_CPU_PCH_ENABLE};NGCPU_DEBUGINFO=${NGRAPH_CPU_DEBUGINFO_ENABLE}")
endif()

add_library(ngraph SHARED ${SRC})

if (NOT APPLE)
    # NGraph links against one or more libraries (ex. LLVM) but we don't want to
    # export these symbols as part of the DSO. This is a GNU ld (and derivatives) specific
    # option so making this portable is still an open issue. As a note for the future,
    # this is not an issue on Windows and LLVM's lld does support --exclude-libs.
    set_target_properties(ngraph PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
endif()

# Colon separated string for specified runtime plugin loading, this is made explicit s.t. if a
# plugin is specified at compile time but the corresponding library could not be resolved at run-
# time, an error will be generated.
# E.g. assume compiling with Argon and Xpu, then -DRUNTIME_PLUGIN_LIBS="libargon.so:libxpu.so".
if (DEFINED RUNTIME_PLUGIN_LIBS)
    target_compile_definitions(ngraph PRIVATE RUNTIME_PLUGIN_LIBS=${RUNTIME_PLUGIN_LIBS})
else()
    target_compile_definitions(ngraph PRIVATE RUNTIME_PLUGIN_LIBS="")
endif()

# This is used to ensure that libngraph.so and libargon.so are in the same directory for dlopen.
# Effective at build time. Does not affect `make install` logics.
if (DEFINED COMMON_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(ngraph PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${COMMON_LIBRARY_OUTPUT_DIRECTORY})
else()
    set_target_properties(ngraph PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
message(STATUS "LIBRARY_OUTPUT_DIRECTORY set to: ${COMMON_LIBRARY_OUTPUT_DIRECTORY}")


target_include_directories(ngraph PUBLIC "${NGRAPH_INCLUDE_PATH}")

if(LLVM_LINK_LIBS)
    target_link_libraries(ngraph PRIVATE ${LLVM_LINK_LIBS})
endif()

if (APPLE)
    set_property(TARGET ngraph PROPERTY PREFIX "lib")
    set_property(TARGET ngraph PROPERTY OUTPUT_NAME "ngraph.so")
    set_property(TARGET ngraph PROPERTY SUFFIX "")
endif()

if(MKLDNN_LIB_DIR)
    target_link_libraries(ngraph PRIVATE mkldnn)
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
if (LLVM_INCLUDE_DIR AND
    MKLDNN_INCLUDE_DIR)
    # Currently this is required for Argon codegen. To avoid this, we'll need to decouple LLVM from
    # compiler.hpp
    install(DIRECTORY
        ${LLVM_INCLUDE_DIR}/
        DESTINATION "${NGRAPH_INSTALL_INCLUDE}"
    )
    install(DIRECTORY
        ${LLVM_LIB_DIR}/
        DESTINATION "${NGRAPH_INSTALL_LIB}"
)
endif()

add_dependencies(ngraph eigen)

if(LLVM_INCLUDE_DIR)
    add_dependencies(ngraph ext_llvm)
endif()

if(MKLDNN_INCLUDE_DIR)
    add_dependencies(ngraph ext_mkldnn)
endif()
