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

if (NGRAPH_CPU_ENABLE AND (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    message(FATAL_ERROR "The NGRAPH_USE_PREBUILT_LLVM option is not supported on this platform.")
endif()

if (NGRAPH_CPU_ENABLE)
    message(STATUS "Fetching LLVM from llvm.org")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        # Override default LLVM binaries
        if(NOT DEFINED LLVM_TARBALL_URL)
            set(LLVM_TARBALL_URL http://releases.llvm.org/5.0.1/clang+llvm-5.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
        endif()

        if(NOT DEFINED LLVM_SHA1_HASH)
            set(LLVM_SHA1_HASH 2fddf9a90b182fa594786be6923e58f5ead71e9c)
        endif()
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        # Override default LLVM binaries
        if(NOT DEFINED LLVM_TARBALL_URL)
            set(LLVM_TARBALL_URL http://releases.llvm.org/5.0.1/clang+llvm-5.0.1-x86_64-apple-darwin.tar.xz)
        endif()

        if(NOT DEFINED LLVM_SHA1_HASH)
            set(LLVM_SHA1_HASH c8d6f64eab0074cd7519266ffe6663922be4105f)
        endif()
    else()
        message(FATAL_ERROR "The NGRAPH_USE_PREBUILT_LLVM option is not supported on this platform.")
    endif()
    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
    if(${CMAKE_VERSION} VERSION_LESS 3.2)
        ExternalProject_Add(
            ext_llvm
            URL ${LLVM_TARBALL_URL}
            URL_HASH SHA1=${LLVM_SHA1_HASH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            )
    else()
        ExternalProject_Add(
            ext_llvm
            URL ${LLVM_TARBALL_URL}
            URL_HASH SHA1=${LLVM_SHA1_HASH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/ext_llvm-prefix/src/ext_llvm/lib/libLLVMCore.a"
            )
    endif()

    ExternalProject_Get_Property(ext_llvm source_dir)
    set(LLVM_INCLUDE_DIR "${source_dir}/include" PARENT_SCOPE)
    set(LLVM_INCLUDE_DIR "${source_dir}/include")  # used by other external projects in current scope
    set(LLVM_LIB_DIR "${source_dir}/lib" PARENT_SCOPE)

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
