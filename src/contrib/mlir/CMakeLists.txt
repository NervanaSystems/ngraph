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

include_directories(
    ${NGRAPH_INCLUDE_PATH}
    ${MLIR_LLVM_INCLUDE_PATH}
    ${MLIR_INCLUDE_PATHS}
)

add_subdirectory(tools/ngraph-opt)

set(SRC
    backend/cpu/cpu_backend.cpp
    backend/pass/affine_lowerer.cpp
    backend/pass/memory_optimization.cpp
    core/compiler.cpp
    core/ngraph_dialect/dialect.cpp
    core/ngraph_dialect/type.cpp
    core/ngraph_dialect/ops.cpp
    core/pass/mlir_subgraph_extraction.cpp
    core/pass/mlir_subgraph_extraction.hpp
    core/pass/ng_dialect_builder.cpp
    core/pass/ng_dialect_builder.hpp
    runtime/cpu/memory_manager.cpp
    runtime/cpu/cpu_runtime.cpp
    utils.cpp
)

add_library(mlir_backend SHARED ${SRC})

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link MLIR libs
target_link_libraries(
    mlir_backend PRIVATE 
    MLIRAffineToStandard
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRStandardToLLVM
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRSupport
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
whole_archive_link(mlir_backend ${LIBS})
# Link LLVM libs
target_link_libraries(
    mlir_backend PRIVATE 
    ${llvm_libs}
)

# Link ngraph 
target_link_libraries(mlir_backend PUBLIC ngraph)

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

add_dependencies(mlir_backend ngraph_ops_gen ngraph_ops_interfaces_gen ngraph_ops_attributes_gen)

target_include_directories(mlir_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS mlir_backend DESTINATION ${NGRAPH_INSTALL_LIB})
