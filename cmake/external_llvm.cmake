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

include(ExternalProject)

if((NGRAPH_CPU_ENABLE OR NGRAPH_GPU_ENABLE) AND (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    set(CMAKE_DISABLE_SOURCE_CHANGES ON)
    set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

    ExternalProject_Add(
        ext_clang
        GIT_REPOSITORY https://github.com/llvm-mirror/clang.git
        GIT_TAG 26cac19a0d622afc91cd52a002921074bccc6a27
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/clang/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/clang/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/clang/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/clang/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/clang/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/clang"
    )

    ExternalProject_Get_Property(ext_clang SOURCE_DIR)
    set(CLANG_SOURCE_DIR ${SOURCE_DIR})

    ExternalProject_Add(
        ext_openmp
        GIT_REPOSITORY https://github.com/llvm-mirror/openmp.git
        GIT_TAG 29b515e1e6d26b5b0d32d47d28dcdb4b8a11470d
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/openmp"
    )

    ExternalProject_Get_Property(ext_openmp SOURCE_DIR)
    set(OPENMP_SOURCE_DIR ${SOURCE_DIR})

    if(DEFINED CMAKE_ASM_COMPILER)
        set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_ASM_COMPILER})
    else()
        set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
    endif()

    ExternalProject_Add(
        ext_llvm
        DEPENDS ext_clang ext_openmp
        GIT_REPOSITORY https://github.com/llvm-mirror/llvm.git
        GIT_TAG 5ae73c34f7eca6c43e71038b06704a8f7abc7f26
        CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_ASM_COMPILER=${LLVM_CMAKE_ASM_COMPILER}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/llvm
                   -DCMAKE_BUILD_TYPE=Release
                   -DLLVM_ENABLE_ASSERTIONS=OFF
                   -DLLVM_INCLUDE_TESTS=OFF
                   -DLLVM_INCLUDE_EXAMPLES=OFF
                   -DLLVM_BUILD_TOOLS=ON
                   -DLLVM_TARGETS_TO_BUILD=X86
                   -DLLVM_EXTERNAL_CLANG_SOURCE_DIR=${CLANG_SOURCE_DIR}
                   -DLLVM_EXTERNAL_OPENMP_SOURCE_DIR=${OPENMP_SOURCE_DIR}
        UPDATE_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm"
        BUILD_BYPRODUCTS ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMCore.a
    )

    ExternalProject_Get_Property(ext_llvm INSTALL_DIR)

    set(LLVM_INCLUDE_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/include")
    set(LLVM_LIB_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/lib/")

    if(APPLE)
        set(LLVM_LINK_LIBS
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangTooling
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangFrontendTool
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangSerialization
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangCodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangParse
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangSema
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerCheckers
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerCore
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangAnalysis
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangARCMigrate
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangRewriteFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangEdit
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangAST
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangLex
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangBasic
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLTO
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMPasses
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObjCARCOpts
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSymbolize
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoPDB
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoDWARF
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMIRParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCoverage
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTableGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDlltoolDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMOrcJIT
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObjectYAML
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLibDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMOption
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Disassembler
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86AsmParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86CodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMGlobalISel
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSelectionDAG
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAsmPrinter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoCodeView
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoMSF
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Desc
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCDisassembler
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Info
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86AsmPrinter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Utils
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCJIT
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLineEditor
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInterpreter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMExecutionEngine
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMRuntimeDyld
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTarget
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCoroutines
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMipo
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInstrumentation
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMVectorize
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMScalarOpts
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLinker
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMIRReader
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAsmParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInstCombine
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTransformUtils
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBitWriter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAnalysis
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMProfileData
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObject
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMC
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBitReader
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCore
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBinaryFormat
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSupport
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDemangle
            curses
            z
            m
        )
    else()
        set(LLVM_LINK_LIBS
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangTooling
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangFrontendTool
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangSerialization
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangCodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangParse
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangSema
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerCheckers
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangStaticAnalyzerCore
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangAnalysis
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangARCMigrate
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangRewriteFrontend
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangEdit
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangAST
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangLex
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/clangBasic
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLTO
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMPasses
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObjCARCOpts
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSymbolize
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoPDB
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoDWARF
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMIRParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCoverage
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTableGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDlltoolDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMOrcJIT
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObjectYAML
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLibDriver
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMOption
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Disassembler
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86AsmParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86CodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMGlobalISel
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSelectionDAG
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAsmPrinter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoCodeView
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDebugInfoMSF
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Desc
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCDisassembler
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Info
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86AsmPrinter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMX86Utils
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCJIT
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLineEditor
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInterpreter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMExecutionEngine
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMRuntimeDyld
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCodeGen
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTarget
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCoroutines
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMipo
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInstrumentation
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMVectorize
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMScalarOpts
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMLinker
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMIRReader
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAsmParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMInstCombine
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMTransformUtils
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBitWriter
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMAnalysis
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMProfileData
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMObject
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMCParser
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMMC
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBitReader
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMCore
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMBinaryFormat
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMSupport
            ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/LLVMDemangle
            tinfo
            z
            m
        )
    endif()
endif()
