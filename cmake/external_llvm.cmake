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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        set(COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    endif()
endif()

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
                -DCMAKE_CXX_FLAGS=${COMPILE_FLAGS}
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
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangTooling${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangFrontendTool${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangSerialization${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangCodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangParse${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangSema${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerCheckers${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerCore${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangAnalysis${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangARCMigrate${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangRewriteFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangEdit${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangAST${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangLex${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangBasic${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLTO${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMPasses${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObjCARCOpts${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSymbolize${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoPDB${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoDWARF${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMIRParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCoverage${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTableGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDlltoolDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMOrcJIT${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObjectYAML${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLibDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMOption${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Disassembler${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86AsmParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86CodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMGlobalISel${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSelectionDAG${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAsmPrinter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoCodeView${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoMSF${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Desc${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCDisassembler${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Info${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86AsmPrinter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Utils${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCJIT${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLineEditor${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInterpreter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMExecutionEngine${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMRuntimeDyld${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTarget${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCoroutines${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMipo${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInstrumentation${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMVectorize${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMScalarOpts${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLinker${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMIRReader${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAsmParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInstCombine${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTransformUtils${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBitWriter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAnalysis${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMProfileData${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObject${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMC${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBitReader${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCore${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBinaryFormat${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSupport${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${EXTERNAL_PROJECTS_ROOT}/llvm/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDemangle${CMAKE_STATIC_LIBRARY_SUFFIX}
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
