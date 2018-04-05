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

    ExternalProject_Add(ext_clang
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

    ExternalProject_Add(ext_openmp
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

    ExternalProject_Add(ext_llvm
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

    set(LLVM_INCLUDE_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/include" PARENT_SCOPE)
    set(LLVM_INCLUDE_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/include")  # used by other external projects in current scope
    set(LLVM_LIB_DIR "${EXTERNAL_PROJECTS_ROOT}/llvm/lib" PARENT_SCOPE)

    set(LLVM_COMMON_LINK_LIBS
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
        z
        m)

    if(APPLE)
        set(LLVM_LINK_LIBS
            ${LLVM_COMMON_LINK_LIBS}
            curses
            PARENT_SCOPE)
    else()
        set(LLVM_LINK_LIBS
            ${LLVM_COMMON_LINK_LIBS}
            tinfo
            PARENT_SCOPE)
    endif()
endif()
