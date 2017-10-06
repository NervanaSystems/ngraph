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

find_package(LLVM CONFIG)

if(LLVM_FOUND)
    if(${LLVM_PACKAGE_VERSION} VERSION_GREATER "4.0")
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}") 
    else()
        set(LLVM_FOUND FALSE)
    endif()
endif()

if(NOT LLVM_FOUND)
    if((NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin") AND
       (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
        message(STATUS "Fetching LLVM from llvm.org")
        set(LLVM_RELEASE_URL http://releases.llvm.org/5.0.0/clang+llvm-5.0.0-linux-x86_64-ubuntu16.04.tar.xz)

        # Override default LLVM binaries
        if(PREBUILT_LLVM)
            set(LLVM_RELEASE_URL ${PREBUILT_LLVM})
        endif()

        # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
        if(${CMAKE_VERSION} VERSION_LESS 3.2)
            ExternalProject_Add(
                ext_llvm
                URL ${LLVM_RELEASE_URL}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
                UPDATE_COMMAND ""
                )
        else()
            ExternalProject_Add(
                ext_llvm
                URL ${LLVM_RELEASE_URL}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
                UPDATE_COMMAND ""
                BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/ext_llvm-prefix/src/ext_llvm/lib/libLLVMCore.a"
                )
        endif()

        ExternalProject_Get_Property(ext_llvm source_dir)
        set(LLVM_INCLUDE_DIR "${source_dir}/include")
        set(LLVM_LIB_DIR "${source_dir}/lib")

        set(LLVM_LINK_LIBS clangTooling clangFrontendTool clangFrontend clangDriver clangSerialization clangCodeGen clangParse clangSema clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers clangStaticAnalyzerCore clangAnalysis clangARCMigrate clangRewriteFrontend clangEdit clangAST clangLex clangBasic LLVMLTO LLVMPasses LLVMObjCARCOpts LLVMSymbolize LLVMDebugInfoPDB LLVMDebugInfoDWARF LLVMMIRParser LLVMCoverage LLVMTableGen LLVMDlltoolDriver LLVMOrcJIT LLVMXCoreDisassembler LLVMXCoreCodeGen LLVMXCoreDesc LLVMXCoreInfo LLVMXCoreAsmPrinter LLVMSystemZDisassembler LLVMSystemZCodeGen LLVMSystemZAsmParser LLVMSystemZDesc LLVMSystemZInfo LLVMSystemZAsmPrinter LLVMSparcDisassembler LLVMSparcCodeGen LLVMSparcAsmParser LLVMSparcDesc LLVMSparcInfo LLVMSparcAsmPrinter LLVMPowerPCDisassembler LLVMPowerPCCodeGen LLVMPowerPCAsmParser LLVMPowerPCDesc LLVMPowerPCInfo LLVMPowerPCAsmPrinter LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo LLVMNVPTXAsmPrinter LLVMMSP430CodeGen LLVMMSP430Desc LLVMMSP430Info LLVMMSP430AsmPrinter LLVMMipsDisassembler LLVMMipsCodeGen LLVMMipsAsmParser LLVMMipsDesc LLVMMipsInfo LLVMMipsAsmPrinter LLVMLanaiDisassembler LLVMLanaiCodeGen LLVMLanaiAsmParser LLVMLanaiDesc LLVMLanaiAsmPrinter LLVMLanaiInfo LLVMHexagonDisassembler LLVMHexagonCodeGen LLVMHexagonAsmParser LLVMHexagonDesc LLVMHexagonInfo LLVMBPFDisassembler LLVMBPFCodeGen LLVMBPFDesc LLVMBPFInfo LLVMBPFAsmPrinter LLVMARMDisassembler LLVMARMCodeGen LLVMARMAsmParser LLVMARMDesc LLVMARMInfo LLVMARMAsmPrinter LLVMAMDGPUDisassembler LLVMAMDGPUCodeGen LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUInfo LLVMAMDGPUAsmPrinter LLVMAMDGPUUtils LLVMAArch64Disassembler LLVMAArch64CodeGen LLVMAArch64AsmParser LLVMAArch64Desc LLVMAArch64Info LLVMAArch64AsmPrinter LLVMAArch64Utils LLVMObjectYAML LLVMLibDriver LLVMOption LLVMX86Disassembler LLVMX86AsmParser LLVMX86CodeGen LLVMGlobalISel LLVMSelectionDAG LLVMAsmPrinter LLVMDebugInfoCodeView LLVMDebugInfoMSF LLVMX86Desc LLVMMCDisassembler LLVMX86Info LLVMX86AsmPrinter LLVMX86Utils LLVMMCJIT LLVMLineEditor LLVMInterpreter LLVMExecutionEngine LLVMRuntimeDyld LLVMCodeGen LLVMTarget LLVMCoroutines LLVMipo LLVMInstrumentation LLVMVectorize LLVMScalarOpts LLVMLinker LLVMIRReader LLVMAsmParser LLVMInstCombine LLVMTransformUtils LLVMBitWriter LLVMAnalysis LLVMProfileData LLVMObject LLVMMCParser LLVMMC LLVMBitReader LLVMCore LLVMBinaryFormat LLVMSupport LLVMDemangle tinfo z m)
        set(LLVM_FOUND TRUE)
        set(Clang_FOUND TRUE)
    endif()
endif()

if(LLVM_FOUND AND NOT Clang_FOUND)
    find_package(Clang CONFIG)
endif()

# TODO: Figure out if this terminates the build or do we allow interpretation-only builds
#if(NOT LLVM_FOUND OR NOT Clang_FOUND)
#endif()

# Populate header and library paths from package-exported info
# if we found system-level LLVM and Clang packages
if(NOT LLVM_INCLUDE_DIR)
    set(LLVM_INCLUDE_DIR ${LLVM_INCLUDE_DIRS})
    set(LLVM_LIB_DIR ${LLVM_LIBRARY_DIRS})
    llvm_map_components_to_libnames(llvm_libs support core engine)
    set(LLVM_LINK_LIBS ${llvm_libs})
endif()

# Export all necessary info
set(LLVM_INCLUDE_DIR ${LLVM_INCLUDE_DIR} PARENT_SCOPE)
set(LLVM_LIB_DIR ${LLVM_LIB_DIR} PARENT_SCOPE)
set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} PARENT_SCOPE)
