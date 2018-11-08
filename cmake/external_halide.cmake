# ******************************************************************************
# Copyright 2018 Intel Corporation
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

find_package(ZLIB REQUIRED)

set(HALIDE_LLVM_TARBALL_URL https://releases.llvm.org/6.0.1/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
set(HALIDE_LLVM_SHA1_HASH c7db0162fbf4cc32193b6a85f84f4abee3d107b9)

ExternalProject_Add(
    ext_halide_llvm
    URL ${HALIDE_LLVM_TARBALL_URL}
    URL_HASH SHA1=${HALIDE_LLVM_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_halide_llvm SOURCE_DIR)

set(HALIDE_LLVM_DIR "${SOURCE_DIR}/lib/cmake/llvm")

set(HALIDE_GIT_REPO_URL https://github.com/halide/Halide)
set(HALIDE_GIT_TAG "ea9c863")

ExternalProject_Add(
    ext_halide
    DEPENDS ext_halide_llvm
    GIT_REPOSITORY ${HALIDE_GIT_REPO_URL}
    GIT_TAG ${HALIDE_GIT_TAG}
    UPDATE_COMMAND ""
    CMAKE_ARGS
    -DLLVM_DIR=${HALIDE_LLVM_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/halide
    -DHALIDE_SHARED_LIBRARY=OFF
    -DWITH_APPS=OFF
    -DWITH_TUTORIALS=OFF
    -DWITH_TESTS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/halide/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/halide/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/halide/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/halide/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/halide/build"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/halide"
    EXCLUDE_FROM_ALL TRUE
    )

# Not sure if all of these are used by Halide but we can trim it down later
# if needed
set(HALIDE_LLVM_LINK_LIBS
    ${SOURCE_DIR}/lib/libLLVMX86AsmParser.a
    ${SOURCE_DIR}/lib/libLLVMX86CodeGen.a
    ${SOURCE_DIR}/lib/libLLVMGlobalISel.a
    ${SOURCE_DIR}/lib/libLLVMSelectionDAG.a
    ${SOURCE_DIR}/lib/libLLVMAsmPrinter.a
    ${SOURCE_DIR}/lib/libLLVMDebugInfoCodeView.a
    ${SOURCE_DIR}/lib/libLLVMX86Desc.a
    ${SOURCE_DIR}/lib/libLLVMMCDisassembler.a
    ${SOURCE_DIR}/lib/libLLVMX86Info.a
    ${SOURCE_DIR}/lib/libLLVMX86AsmPrinter.a
    ${SOURCE_DIR}/lib/libLLVMX86Utils.a
    ${SOURCE_DIR}/lib/libLLVMMCJIT.a
    ${SOURCE_DIR}/lib/libLLVMLineEditor.a
    ${SOURCE_DIR}/lib/libLLVMInterpreter.a
    ${SOURCE_DIR}/lib/libLLVMExecutionEngine.a
    ${SOURCE_DIR}/lib/libLLVMRuntimeDyld.a
    ${SOURCE_DIR}/lib/libLLVMCodeGen.a
    ${SOURCE_DIR}/lib/libLLVMTarget.a
    ${SOURCE_DIR}/lib/libLLVMCoroutines.a
    ${SOURCE_DIR}/lib/libLLVMipo.a
    ${SOURCE_DIR}/lib/libLLVMInstrumentation.a
    ${SOURCE_DIR}/lib/libLLVMVectorize.a
    ${SOURCE_DIR}/lib/libLLVMScalarOpts.a
    ${SOURCE_DIR}/lib/libLLVMLinker.a
    ${SOURCE_DIR}/lib/libLLVMIRReader.a
    ${SOURCE_DIR}/lib/libLLVMAsmParser.a
    ${SOURCE_DIR}/lib/libLLVMInstCombine.a
    ${SOURCE_DIR}/lib/libLLVMTransformUtils.a
    ${SOURCE_DIR}/lib/libLLVMBitWriter.a
    ${SOURCE_DIR}/lib/libLLVMAnalysis.a
    ${SOURCE_DIR}/lib/libLLVMProfileData.a
    ${SOURCE_DIR}/lib/libLLVMObject.a
    ${SOURCE_DIR}/lib/libLLVMMCParser.a
    ${SOURCE_DIR}/lib/libLLVMMC.a
    ${SOURCE_DIR}/lib/libLLVMBitReader.a
    ${SOURCE_DIR}/lib/libLLVMCore.a
    ${SOURCE_DIR}/lib/libLLVMBinaryFormat.a
    ${SOURCE_DIR}/lib/libLLVMSupport.a
)

add_library(libhalidellvm INTERFACE)
add_dependencies(libhalidellvm ext_halide_llvm)
target_include_directories(libhalidellvm SYSTEM INTERFACE ${EXTERNAL_PROJECTS_ROOT}/halide_llvm/include)
target_link_libraries(libhalidellvm INTERFACE ${HALIDE_LLVM_LINK_LIBS})
