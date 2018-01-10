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
    builder/autobroadcast.cpp
    builder/numpy_transpose.cpp
    builder/reduce_ops.cpp
    builder/xla_tuple.cpp
    common.cpp
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
    ops/binary_elementwise_arithmetic.cpp
    ops/binary_elementwise_comparison.cpp
    ops/binary_elementwise.cpp
    ops/broadcast.cpp
    ops/concatenate.cpp
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
    ops/parameter.cpp
    ops/power.cpp
    ops/reduce.cpp
    ops/reduce_window.cpp
    ops/replace_slice.cpp
    ops/reshape.cpp
    ops/reverse.cpp
    ops/select.cpp
    ops/select_and_scatter.cpp
    ops/sin.cpp
    ops/sinh.cpp
    ops/slice.cpp
    ops/sqrt.cpp
    ops/subtract.cpp
    ops/sum.cpp
    ops/tan.cpp
    ops/tanh.cpp
    ops/unary_elementwise_arithmetic.cpp
    ops/unary_elementwise.cpp
    pass/dump_sorted.cpp
    pass/graph_rewrite.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/pass.cpp
    pass/visualize_tree.cpp
    pattern/matcher.cpp
    runtime/aligned_buffer.cpp
    runtime/interpreter/int_backend.cpp
    runtime/interpreter/int_call_frame.cpp
    runtime/interpreter/int_external_function.cpp
    runtime/interpreter/int_kernels.cpp
    runtime/interpreter/int_manager.cpp
    runtime/interpreter/int_tensor_view.cpp
    runtime/manager.cpp
    runtime/tensor_view.cpp
    serializer.cpp
    shape.cpp
    types/element_type.cpp
    types/type.cpp
    util.cpp
    graph_util.cpp
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

include_directories(
    "${NGRAPH_INCLUDE_PATH}"
    )

include_directories(
    SYSTEM
    "${EIGEN_INCLUDE_DIR}"
    )

if (NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR AND
    MKLDNN_INCLUDE_DIR)
    find_package(ZLIB REQUIRED)

    include(${TBB_ROOT}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS compiler=clang tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/tbb_build
              tbb_build_prefix=tbb CONFIG_DIR TBB_DIR)
    find_package(TBB REQUIRED tbb)
    if (NOT TBB_FOUND)
        message(FATAL_ERROR "TBB is needed by the CPU backend and was not found")
    else()
        message(STATUS "Found TBB and imported target ${TBB_IMPORTED_TARGETS}")
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
    )
    # LLVM binary builds are typically built without RTTI
    # The built-in headers are in a version-specific directory
    # This must be kept in sync with the LLVM + Clang version in use
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
    set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_DEFINITIONS
      "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";TBB_HEADERS_PATH=\"${TBB_ROOT}/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";INSTALLED_HEADERS_PATH=\"${CMAKE_INSTALL_PREFIX}/include\";")
    set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")
    set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
      "NGCPU_DEBUGINFO=${NGRAPH_CPU_DEBUGINFO_ENABLE};")

if (NGRAPH_ARGON_ENABLE)
    link_directories(${ARGON_LIB_DIR})
endif()

    # GPU backend current requires CPU because they share compiler.cpp,
    # and compiler.cpp requires MKLDNN
    if(NGRAPH_GPU_ENABLE)
        include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
        link_directories(${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})

        # Add sources for the GPU backend
        # and all its dependencies
        set(SRC ${SRC}
            runtime/gpu/gpu_call_frame.cpp
            runtime/gpu/gpu_backend.cpp
            runtime/gpu/gpu_manager.cpp
            runtime/gpu/gpu_external_function.cpp
            runtime/gpu/gpu_tensor_view.cpp
            )
        set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
        "CUDA_HEADER_PATHS=\"${CUDA_INCLUDE_DIRS}\";")
    endif()
endif()

add_library(ngraph SHARED ${SRC})
if (NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR AND MKLDNN_INCLUDE_DIR)
    # Generate the resource file containing all headers used by the codegen compiler
    add_custom_target(header_resource
        resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
        DEPENDS resource_generator eigen ext_llvm ext_mkldnn
        BYPRODUCTS
    )
    add_dependencies(ngraph header_resource)
    include_directories(${CMAKE_BINARY_DIR})
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

# Colon separated string for specified runtime plugin loading, this is made explicit s.t. if a
# plugin is specified at compile time but the corresponding library could not be resolved at run-
# time, an error will be generated.
# E.g. assume compiling with Argon and Xpu, then -DRUNTIME_PLUGIN_LIBS="libargon.so:libxpu.so".

if (NGRAPH_ARGON_ENABLE)
    set(RUNTIME_PLUGIN_LIBS "libargon.so")  # TODO: currently hardcoded argon lib name if enabled
endif()

if (DEFINED RUNTIME_PLUGIN_LIBS)
    target_compile_definitions(ngraph PRIVATE RUNTIME_PLUGIN_LIBS="${RUNTIME_PLUGIN_LIBS}")
else()
    target_compile_definitions(ngraph PRIVATE RUNTIME_PLUGIN_LIBS="")
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

if(NGRAPH_GPU_ENABLE AND CUDA_LIBRARIES)
    target_link_libraries(ngraph PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})
endif()

# Argon
if (NGRAPH_ARGON_ENABLE)
    target_link_libraries(ngraph PRIVATE argon)
    add_dependencies(ngraph ext_argon)
    add_dependencies(ext_argon ext_llvm)
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

if (NOT APPLE)
    install(DIRECTORY
        ${MKLDNN_LIB_DIR}/
        DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
    if (NGRAPH_CPU_ENABLE)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/tbb_release/
            DESTINATION ${NGRAPH_INSTALL_LIB}
            FILES_MATCHING PATTERN "libtbb.so.*"
        )
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/tbb_debug/
            DESTINATION ${NGRAPH_INSTALL_LIB}
            FILES_MATCHING PATTERN "libtbb_debug.so.*"
        )
    endif()
endif()

