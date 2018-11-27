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

#------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#------------------------------------------------------------------------------

if(MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
    ExternalProject_Add(
        ext_mkldnn
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        )
    add_library(libmkldnn INTERFACE)
    target_include_directories(libmkldnn SYSTEM INTERFACE ${MKLDNN_INCLUDE_DIR})
    target_link_libraries(libmkldnn INTERFACE
        ${MKLDNN_LIB_DIR}/libmkldnn.so
        ${MKLDNN_LIB_DIR}/libmklml_intel.so
        ${MKLDNN_LIB_DIR}/libiomp5.so
        )

    install(DIRECTORY ${MKLDNN_LIB_DIR}/ DESTINATION ${NGRAPH_INSTALL_LIB})
    return()
endif()

# This section sets up MKL as an external project to be used later by MKLDNN

set(MKLURLROOT "https://github.com/intel/mkl-dnn/releases/download/v0.17/")
set(MKLVERSION "2019.0.1.20180928")
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(MKLPACKAGE "mklml_lnx_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 0d9cc8bfc2c1a1e3df5e0b07e2f363bbf934a7e9)
    set(MKL_LIBS libiomp5.so libmklml_intel.so)
elseif (APPLE)
    set(MKLPACKAGE "mklml_mac_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 232787f41cb42f53f5b7e278d8240f6727896133)
    set(MKL_LIBS libmklml.dylib libiomp5.dylib)
elseif (WIN32)
    set(MKLPACKAGE "mklml_win_${MKLVERSION}.zip")
    set(MKL_SHA1_HASH 97f01ab854d8ee88cc0429f301df84844d7cce6b)
    set(MKL_LIBS mklml.dll libiomp5md.dll)
endif()
set(MKLURL ${MKLURLROOT}${MKLPACKAGE})

ExternalProject_Add(
    ext_mkl
    PREFIX mkl
    URL ${MKLURL}
    URL_HASH SHA1=${MKL_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
)
ExternalProject_Get_Property(ext_mkl source_dir)
set(MKL_ROOT ${EXTERNAL_PROJECTS_ROOT}/mkldnn/src/external/mkl)
set(MKL_SOURCE_DIR ${source_dir})
add_library(libmkl INTERFACE)
add_dependencies(libmkl ext_mkl)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/${LIB})
endforeach()

set(MKLDNN_GIT_REPO_URL https://github.com/intel/mkl-dnn)
set(MKLDNN_GIT_TAG "830a100")
if(NGRAPH_LIB_VERSIONING_ENABLE)
    set(MKLDNN_PATCH_FILE mkldnn.patch)
else()
    set(MKLDNN_PATCH_FILE mkldnn_no_so_link.patch)
endif()

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if(${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_mkldnn
        DEPENDS ext_mkl
        GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
        GIT_TAG ${MKLDNN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        # Patch gets mad if it applied for a second time so:
        #    --forward tells patch to ignore if it has already been applied
        #    --reject-file tells patch to not right a reject file
        #    || exit 0 changes the exit code for the PATCH_COMMAND to zero so it is not an error
        # I don't like it, but it works
        PATCH_COMMAND patch -p1 --forward --reject-file=- -i ${CMAKE_SOURCE_DIR}/cmake/${MKLDNN_PATCH_FILE} || exit 0
        # Uncomment below with any in-flight MKL-DNN patches
        # PATCH_COMMAND patch -p1 < ${CMAKE_SOURCE_DIR}/third-party/patches/mkldnn-cmake-openmp.patch
        CMAKE_ARGS
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/mkldnn
            -DMKLROOT=${MKL_ROOT}
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn"
        EXCLUDE_FROM_ALL TRUE
        )
else()
    ExternalProject_Add(
        ext_mkldnn
        DEPENDS ext_mkl
        GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
        GIT_TAG ${MKLDNN_GIT_TAG}
        UPDATE_COMMAND ""
        # Patch gets mad if it applied for a second time so:
        #    --forward tells patch to ignore if it has already been applied
        #    --reject-file tells patch to not right a reject file
        #    || exit 0 changes the exit code for the PATCH_COMMAND to zero so it is not an error
        # I don't like it, but it works
        PATCH_COMMAND patch -p1 --forward --reject-file=- -i ${CMAKE_SOURCE_DIR}/cmake/${MKLDNN_PATCH_FILE} || exit 0
        # Uncomment below with any in-flight MKL-DNN patches
        # PATCH_COMMAND patch -p1 < ${CMAKE_SOURCE_DIR}/third-party/patches/mkldnn-cmake-openmp.patch
        CMAKE_ARGS
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/mkldnn
            -DMKLROOT=${MKL_ROOT}
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn"
        BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/mkldnn/include/mkldnn.hpp"
        EXCLUDE_FROM_ALL TRUE
        )
endif()

ExternalProject_Add_Step(
    ext_mkldnn
    PrepareMKL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MKL_SOURCE_DIR} ${MKL_ROOT}
    DEPENDEES download
    DEPENDERS configure
    )

add_custom_command(TARGET ext_mkldnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib ${NGRAPH_BUILD_DIR}
    COMMENT "Move mkldnn libraries to ngraph build directory"
)

add_library(libmkldnn INTERFACE)
add_dependencies(libmkldnn ext_mkldnn)
target_include_directories(libmkldnn SYSTEM INTERFACE ${EXTERNAL_PROJECTS_ROOT}/mkldnn/include)
target_link_libraries(libmkldnn INTERFACE
    ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/libmkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}
    libmkl
    )

install(DIRECTORY ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/ DESTINATION ${NGRAPH_INSTALL_LIB} OPTIONAL)
