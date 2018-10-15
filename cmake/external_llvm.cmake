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
    EXCLUDE_FROM_ALL TRUE
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
    EXCLUDE_FROM_ALL TRUE
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
    GIT_TAG da4a2839d80ac52958be0129b871beedfe90136e
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
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_llvm INSTALL_DIR)

set(LLVM_LINK_LIBS
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangTooling.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangFrontendTool.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangFrontend.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangDriver.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangSerialization.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangCodeGen.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangParse.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangSema.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangStaticAnalyzerFrontend.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangStaticAnalyzerCheckers.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangStaticAnalyzerCore.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangAnalysis.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangARCMigrate.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangRewriteFrontend.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangEdit.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangAST.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangLex.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libclangBasic.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMLTO.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMPasses.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMObjCARCOpts.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMSymbolize.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDebugInfoPDB.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDebugInfoDWARF.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMMIRParser.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMCoverage.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMTableGen.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDlltoolDriver.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMOrcJIT.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMObjectYAML.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMLibDriver.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMOption.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86Disassembler.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86AsmParser.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86CodeGen.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMGlobalISel.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMSelectionDAG.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMAsmPrinter.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDebugInfoCodeView.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDebugInfoMSF.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86Desc.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMMCDisassembler.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86Info.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86AsmPrinter.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMX86Utils.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMMCJIT.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMLineEditor.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMInterpreter.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMExecutionEngine.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMRuntimeDyld.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMCodeGen.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMTarget.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMCoroutines.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMipo.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMInstrumentation.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMVectorize.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMScalarOpts.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMLinker.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMIRReader.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMAsmParser.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMInstCombine.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMTransformUtils.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMBitWriter.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMAnalysis.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMProfileData.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMObject.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMMCParser.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMMC.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMBitReader.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMCore.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMBinaryFormat.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMSupport.a
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/libLLVMDemangle.a
)

if(APPLE)
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} curses z m)
else()
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} tinfo z m)
endif()

add_library(libllvm INTERFACE)
add_dependencies(libllvm ext_llvm)
target_include_directories(libllvm SYSTEM INTERFACE ${EXTERNAL_PROJECTS_ROOT}/llvm/include)
target_link_libraries(libllvm INTERFACE ${LLVM_LINK_LIBS})
