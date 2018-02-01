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

include(ExternalProject)

if((NGRAPH_CPU_ENABLE OR NGRAPH_GPU_ENABLE) AND (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin") AND
                         (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    set(CMAKE_DISABLE_SOURCE_CHANGES ON)
    set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

    set(RELEASE_TAG release_50)

    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ext_llvm-prefix)

    ExternalProject_Add(clang
        GIT_REPOSITORY https://github.com/llvm-mirror/clang.git
        GIT_TAG ${RELEASE_TAG}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
    )

    ExternalProject_Get_Property(clang SOURCE_DIR)

    if(DEFINED CMAKE_ASM_COMPILER)
        set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_ASM_COMPILER})
    else()
        set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
    endif()

    ExternalProject_Add(ext_llvm
        DEPENDS clang
        GIT_REPOSITORY https://github.com/llvm-mirror/llvm.git
        GIT_TAG ${RELEASE_TAG}
        CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_ASM_COMPILER=${LLVM_CMAKE_ASM_COMPILER}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
                   -DLLVM_INCLUDE_TESTS=OFF
                   -DLLVM_INCLUDE_EXAMPLES=OFF
                   -DLLVM_BUILD_TOOLS=ON
                   -DLLVM_TARGETS_TO_BUILD=X86
                   -DLLVM_EXTERNAL_CLANG_SOURCE_DIR:PATH=${SOURCE_DIR}
        UPDATE_COMMAND ""
    )

    ExternalProject_Get_Property(ext_llvm SOURCE_DIR)
    ExternalProject_Get_Property(ext_llvm BINARY_DIR)
    ExternalProject_Get_Property(ext_llvm INSTALL_DIR)
    message("SOURCE_DIR = ${SOURCE_DIR}")
    message("BINARY_DIR = ${BINARY_DIR}")
    message("INSTALL_DIR = ${INSTALL_DIR}")

    set(LLVM_INCLUDE_DIR "${INSTALL_DIR}/include" PARENT_SCOPE)
    set(LLVM_INCLUDE_DIR "${SOURCE_DIR}/include")  # used by other external projects in current scope
    set(LLVM_LIB_DIR "${INSTALL_DIR}/lib" PARENT_SCOPE)

    set(LLVM_LINK_LIBS
        clangTooling
        clangFrontendTool
        clangFrontend
        clangDriver
        clangSerialization
        clangCodeGen
        clangParse
        clangSema
        clangStaticAnalyzerFrontend
        clangStaticAnalyzerCheckers
        clangStaticAnalyzerCore
        clangAnalysis
        clangARCMigrate
        clangRewriteFrontend
        clangEdit
        clangAST
        clangLex
        clangBasic
        LLVMLTO
        LLVMPasses
        LLVMObjCARCOpts
        LLVMSymbolize
        LLVMDebugInfoPDB
        LLVMDebugInfoDWARF
        LLVMMIRParser
        LLVMCoverage
        LLVMTableGen
        LLVMDlltoolDriver
        LLVMOrcJIT
        LLVMObjectYAML
        LLVMLibDriver
        LLVMOption
        LLVMX86Disassembler
        LLVMX86AsmParser
        LLVMX86CodeGen
        LLVMGlobalISel
        LLVMSelectionDAG
        LLVMAsmPrinter
        LLVMDebugInfoCodeView
        LLVMDebugInfoMSF
        LLVMX86Desc
        LLVMMCDisassembler
        LLVMX86Info
        LLVMX86AsmPrinter
        LLVMX86Utils
        LLVMMCJIT
        LLVMLineEditor
        LLVMInterpreter
        LLVMExecutionEngine
        LLVMRuntimeDyld
        LLVMCodeGen
        LLVMTarget
        LLVMCoroutines
        LLVMipo
        LLVMInstrumentation
        LLVMVectorize
        LLVMScalarOpts
        LLVMLinker
        LLVMIRReader
        LLVMAsmParser
        LLVMInstCombine
        LLVMTransformUtils
        LLVMBitWriter
        LLVMAnalysis
        LLVMProfileData
        LLVMObject
        LLVMMCParser
        LLVMMC
        LLVMBitReader
        LLVMCore
        LLVMBinaryFormat
        LLVMSupport
        LLVMDemangle
        tinfo
        z
        m
        PARENT_SCOPE)
endif()
