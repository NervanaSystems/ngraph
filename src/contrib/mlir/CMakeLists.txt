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

set(SRC
    dialect/dialect.cpp
    dialect/type.cpp
    dialect/ops.cpp
    compiler.cpp
    lowerer.cpp
    memory_manager.cpp
    pass/mlir_subgraph_extraction.cpp
    pass/mlir_subgraph_extraction.hpp
)
if (NGRAPH_MLIR_ENABLE)

    add_library(mlir_backend SHARED ${SRC})

    message(STATUS "LLVM Directory: ${LLVM_DIR}")
    # Link LLVM and MLIR
    find_package(LLVM REQUIRED CONFIG)
    
    set(MLIR_LLVM_INCLUDEPATH ${LLVM_INCLUDE_DIRS})

    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")
    add_definitions(${LLVM_DEFINITIONS})
   
    target_include_directories(mlir_backend PRIVATE ${LLVM_INCLUDE_DIRS})
    
    message(STATUS "MLIR Headers at : ${MLIR_INCLUDE_PATHS}")
    message(STATUS "LLVM Headers at : ${MLIR_LLVM_INCLUDEPATH}")

    target_include_directories(mlir_backend PRIVATE ${MLIR_INCLUDE_PATHS}) 
    
    llvm_map_components_to_libnames(llvm_libs support core irreader)
    
    # Link MLIR libs
    target_link_libraries(
        mlir_backend PRIVATE 
        MLIRAnalysis
        MLIREDSC
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMIR
        MLIRParser
        MLIRPass
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRSupport
    )
    # some libs need whole archive linkage because of Globals static initialization
    function(whole_archive_link target)
        if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
            set(link_flags "-Llib -Wl,-all_load ")
            FOREACH(LIB ${ARGN})
                string(CONCAT link_flags ${link_flags} "${LIB}")
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

    whole_archive_link(mlir_backend
        ${LLVM_BUILD_LIBRARY_DIR}/libMLIRAffineOps.a
        ${LLVM_BUILD_LIBRARY_DIR}/libMLIRStandardOps.a
    )
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
    set(LLVM_TARGET_DEFINITIONS dialect/ops.td)

    ngraph_tablegen(ops.h.inc -gen-op-decls)
    ngraph_tablegen(ops.cpp.inc -gen-op-defs)

    add_public_tablegen_target(ngraph_ops_gen)
    add_dependencies(mlir_backend ngraph_ops_gen)
    target_include_directories(mlir_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()
