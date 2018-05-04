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

message(STATUS "Fetching LLVM from llvm.org")

# Override default LLVM binaries
if(NOT DEFINED LLVM_TARBALL_URL)
    set(LLVM_TARBALL_URL http://releases.llvm.org/5.0.1/clang+llvm-5.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
endif()

if(NOT DEFINED LLVM_SHA1_HASH)
    set(LLVM_SHA1_HASH 2fddf9a90b182fa594786be6923e58f5ead71e9c)
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
set(LLVM_INCLUDE_DIR "${source_dir}/include")
set(LLVM_LIB_DIR "${source_dir}/lib")

set(LLVM_LINK_LIBS
    ${source_dir}/lib/libclangTooling.a
    ${source_dir}/lib/libclangFrontendTool.a
    ${source_dir}/lib/libclangFrontend.a
    ${source_dir}/lib/libclangDriver.a
    ${source_dir}/lib/libclangSerialization.a
    ${source_dir}/lib/libclangCodeGen.a
    ${source_dir}/lib/libclangParse.a
    ${source_dir}/lib/libclangSema.a
    ${source_dir}/lib/libclangStaticAnalyzerFrontend.a
    ${source_dir}/lib/libclangStaticAnalyzerCheckers.a
    ${source_dir}/lib/libclangStaticAnalyzerCore.a
    ${source_dir}/lib/libclangAnalysis.a
    ${source_dir}/lib/libclangARCMigrate.a
    ${source_dir}/lib/libclangRewriteFrontend.a
    ${source_dir}/lib/libclangEdit.a
    ${source_dir}/lib/libclangAST.a
    ${source_dir}/lib/libclangLex.a
    ${source_dir}/lib/libclangBasic.a
    ${source_dir}/lib/libLLVMLTO.a
    ${source_dir}/lib/libLLVMPasses.a
    ${source_dir}/lib/libLLVMObjCARCOpts.a
    ${source_dir}/lib/libLLVMSymbolize.a
    ${source_dir}/lib/libLLVMDebugInfoPDB.a
    ${source_dir}/lib/libLLVMDebugInfoDWARF.a
    ${source_dir}/lib/libLLVMMIRParser.a
    ${source_dir}/lib/libLLVMCoverage.a
    ${source_dir}/lib/libLLVMTableGen.a
    ${source_dir}/lib/libLLVMDlltoolDriver.a
    ${source_dir}/lib/libLLVMOrcJIT.a
    ${source_dir}/lib/libLLVMObjectYAML.a
    ${source_dir}/lib/libLLVMLibDriver.a
    ${source_dir}/lib/libLLVMOption.a
    ${source_dir}/lib/libLLVMX86Disassembler.a
    ${source_dir}/lib/libLLVMX86AsmParser.a
    ${source_dir}/lib/libLLVMX86CodeGen.a
    ${source_dir}/lib/libLLVMGlobalISel.a
    ${source_dir}/lib/libLLVMSelectionDAG.a
    ${source_dir}/lib/libLLVMAsmPrinter.a
    ${source_dir}/lib/libLLVMDebugInfoCodeView.a
    ${source_dir}/lib/libLLVMDebugInfoMSF.a
    ${source_dir}/lib/libLLVMX86Desc.a
    ${source_dir}/lib/libLLVMMCDisassembler.a
    ${source_dir}/lib/libLLVMX86Info.a
    ${source_dir}/lib/libLLVMX86AsmPrinter.a
    ${source_dir}/lib/libLLVMX86Utils.a
    ${source_dir}/lib/libLLVMMCJIT.a
    ${source_dir}/lib/libLLVMLineEditor.a
    ${source_dir}/lib/libLLVMInterpreter.a
    ${source_dir}/lib/libLLVMExecutionEngine.a
    ${source_dir}/lib/libLLVMRuntimeDyld.a
    ${source_dir}/lib/libLLVMCodeGen.a
    ${source_dir}/lib/libLLVMTarget.a
    ${source_dir}/lib/libLLVMCoroutines.a
    ${source_dir}/lib/libLLVMipo.a
    ${source_dir}/lib/libLLVMInstrumentation.a
    ${source_dir}/lib/libLLVMVectorize.a
    ${source_dir}/lib/libLLVMScalarOpts.a
    ${source_dir}/lib/libLLVMLinker.a
    ${source_dir}/lib/libLLVMIRReader.a
    ${source_dir}/lib/libLLVMAsmParser.a
    ${source_dir}/lib/libLLVMInstCombine.a
    ${source_dir}/lib/libLLVMTransformUtils.a
    ${source_dir}/lib/libLLVMBitWriter.a
    ${source_dir}/lib/libLLVMAnalysis.a
    ${source_dir}/lib/libLLVMProfileData.a
    ${source_dir}/lib/libLLVMObject.a
    ${source_dir}/lib/libLLVMMCParser.a
    ${source_dir}/lib/libLLVMMC.a
    ${source_dir}/lib/libLLVMBitReader.a
    ${source_dir}/lib/libLLVMCore.a
    ${source_dir}/lib/libLLVMBinaryFormat.a
    ${source_dir}/lib/libLLVMSupport.a
    ${source_dir}/lib/libLLVMDemangle.a
    tinfo
    z
    m
)
