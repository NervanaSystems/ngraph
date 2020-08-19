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
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRSupport
    MLIRAffineTransforms
)
# some libs need whole archive linkage because of Globals static initialization
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(NGRAPH_MLIR_WHOLE_ARCHIVE_PRE "-Wl,-force_load")
else()
    set(NGRAPH_MLIR_WHOLE_ARCHIVE_PRE "-Wl,--whole-archive")
    set(NGRAPH_MLIR_WHOLE_ARCHIVE_POST "-Wl,--no-whole-archive")
endif()

#target_link_libraries(
#    mlir_llvm_backend PRIVATE
#    ${NGRAPH_MLIR_WHOLE_ARCHIVE_PRE}
#    MLIRAffineOps
#    MLIRStandardOps
#    ${NGRAPH_MLIR_WHOLE_ARCHIVE_POST}
#)

# Link LLVM libs
target_link_libraries(
    mlir_llvm_backend PRIVATE
    ${llvm_libs}
)

# Link ngraph
target_link_libraries(mlir_llvm_backend PUBLIC ngraph libmkl DNNL::dnnl)

# table-gen ops.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops.td)
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ngraph_ops_gen)

# table-gen ops_interfaces.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops_interfaces.td)
mlir_tablegen(ops_interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ops_interfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ngraph_ops_interfaces_gen)

# tabel-gen ops attributes.td
set(LLVM_TARGET_DEFINITIONS core/ngraph_dialect/ops_attributes.td)
mlir_tablegen(ops_attributes.h.inc -gen-enum-decls)
mlir_tablegen(ops_attributes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(ngraph_ops_attributes_gen)

# tabel-gen ops fused_ops_pattern.td
set(LLVM_TARGET_DEFINITIONS core/pass/fused_ops_pattern.td)
mlir_tablegen(fused_ops_pattern.h.inc -gen-rewriters)
add_public_tablegen_target(ngraph_ops_pattern_gen)

add_dependencies(mlir_llvm_backend ngraph_ops_gen ngraph_ops_interfaces_gen ngraph_ops_attributes_gen ngraph_ops_pattern_gen)

target_include_directories(mlir_llvm_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS mlir_llvm_backend DESTINATION ${NGRAPH_INSTALL_LIB})
