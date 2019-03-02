# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

# Includes blas 3.8.0 in mkldnn 
set(NGRAPH_MKLDNN_VERSION "v0.18-rc")
set(NGRAPH_MKLDNN_SUB_VERSION "2019.0.3.20190125")
set(NGRAPH_MKLDNN_GIT_TAG "08bd90c")

#------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#------------------------------------------------------------------------------

set(MKLDNN_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX})
if (LINUX)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif (APPLE)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif (WIN32)
    set(MKLDNN_IMPLIB mkldnn${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(MKLML_LIB mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLML_IMPLIB mklml${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(OMP_LIB libiomp5md${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_IMPLIB libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
    if(NOT LINUX)
        message(FATAL_ERROR "Unsupported platform for prebuilt mkl-dnn!")
    endif()
    if(NOT MKLML_LIB_DIR)
        set(MKLML_LIB_DIR ${MKLDNN_LIB_DIR})
    endif()

    add_library(libmkldnn SHARED IMPORTED)
    set_property(TARGET libmkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_LIB})
    target_include_directories(libmkldnn SYSTEM INTERFACE ${MKLDNN_INCLUDE_DIR})

    install(FILES ${MKLDNN_LIB_DIR}/${MKLDNN_LIB} ${MKLML_LIB_DIR}/${MKLML_LIB} ${MKLML_LIB_DIR}/${OMP_LIB}  DESTINATION ${NGRAPH_INSTALL_LIB})
    return()
endif()

# This section sets up MKL as an external project to be used later by MKLDNN

set(MKLURLROOT "https://github.com/intel/mkl-dnn/releases/download/${NGRAPH_MKLDNN_VERSION}/")
set(MKLVERSION ${NGRAPH_MKLDNN_SUB_VERSION})
if (LINUX)
    set(MKLPACKAGE "mklml_lnx_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 968318286897da5ffd225f0851aec18f02b347f8)
elseif (APPLE)
    set(MKLPACKAGE "mklml_mac_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 8ef2f39b65f23d322af7400d261c3ec883b087c6)
elseif (WIN32)
    set(MKLPACKAGE "mklml_win_${MKLVERSION}.zip")
    set(MKL_SHA1_HASH 8383d11b47960e3cd826e2af4b2a7daa9fbd8b68)
endif()
set(MKL_LIBS ${MKLML_LIB} ${OMP_LIB})
set(MKLURL ${MKLURLROOT}${MKLPACKAGE})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(APPLE)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 8.0)
            set(MKLDNN_FLAG "-Wno-stringop-truncation -Wno-stringop-overflow")
        endif()
    elseif(NGRAPH_MANYLINUX_ENABLE) #pragma GCC diagnostic ignored does not work on GCC used for manylinux1
        set(MKLDNN_FLAG "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds")
        set(MKLDNN_FLAG "${MKLDNN_FLAG} -Wno-unused-result -Wno-unused-value")
    endif()
endif()

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

set(MKLDNN_DEPENDS ext_mkl)
ExternalProject_Get_Property(ext_mkl source_dir)
set(MKL_ROOT ${EXTERNAL_PROJECTS_ROOT}/mkldnn/src/external/mkl)
set(MKL_SOURCE_DIR ${source_dir})

ExternalProject_Add_Step(
    ext_mkl
    CopyMKL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${MKLML_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy mklml runtime libraries to ngraph build directory."
    DEPENDEES download
    )

ExternalProject_Add_Step(
    ext_mkl
    CopyOMP
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${OMP_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy OpenMP runtime libraries to ngraph build directory."
    DEPENDEES download
    )

if(WIN32)
    ExternalProject_Add_Step(
        ext_mkl
        CopyMKLIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${MKLML_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}
        COMMENT "Copy mklml runtime libraries to ngraph build directory."
        DEPENDEES download
        )

    ExternalProject_Add_Step(
        ext_mkl
        CopyOMPIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${OMP_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}
        COMMENT "Copy OpenMP runtime libraries to ngraph build directory."
        DEPENDEES download
        )
endif()

add_library(libmkl INTERFACE)
add_dependencies(libmkl ext_mkl)
if(WIN32)
    target_link_libraries(libmkl INTERFACE
        ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLML_IMPLIB}
        ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${OMP_IMPLIB})
else()
    target_link_libraries(libmkl INTERFACE
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${OMP_LIB})
endif()

set(MKLDNN_GIT_REPO_URL https://github.com/intel/mkl-dnn)
set(MKLDNN_GIT_TAG ${NGRAPH_MKLDNN_GIT_TAG})
set(MKLDNN_PATCH_FILE mkldnn.patch)
set(MKLDNN_LIBS ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/${MKLDNN_LIB})

if (WIN32)
    ExternalProject_Add(
        ext_mkldnn
        PREFIX mkldnn
        DEPENDS ${MKLDNN_DEPENDS}
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
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/mkldnn
            -DMKLDNN_ENABLE_CONCURRENT_EXEC=ON
            -DMKLROOT=${MKL_ROOT}
            -DMKLDNN_LIB_VERSIONING_ENABLE=${NGRAPH_LIB_VERSIONING_ENABLE}
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
        PREFIX mkldnn
        DEPENDS ${MKLDNN_DEPENDS}
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
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/mkldnn
            -DMKLDNN_ENABLE_CONCURRENT_EXEC=ON
            -DMKLROOT=${MKL_ROOT}
            -DMKLDNN_LIB_VERSIONING_ENABLE=${NGRAPH_LIB_VERSIONING_ENABLE}
            "-DARCH_OPT_FLAGS=-march=${NGRAPH_TARGET_ARCH} -mtune=${NGRAPH_TARGET_ARCH} ${MKLDNN_FLAG}"
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn"
        EXCLUDE_FROM_ALL TRUE
        )
endif()

ExternalProject_Add_Step(
    ext_mkldnn
    CopyMKLDNN
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/${CMAKE_INSTALL_LIBDIR}/${MKLDNN_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy mkldnn runtime libraries to ngraph build directory."
    DEPENDEES install
    )

if(WIN32)
    ExternalProject_Add_Step(
        ext_mkldnn
        CopyMKLDNNIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/${CMAKE_INSTALL_LIBDIR}/${MKLDNN_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}
        COMMENT "Copy mkldnn runtime libraries to ngraph build directory."
        DEPENDEES install
        )
endif()

ExternalProject_Add_Step(
    ext_mkldnn
    PrepareMKL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MKL_SOURCE_DIR} ${MKL_ROOT}
    DEPENDEES download
    DEPENDERS configure
    )

add_library(libmkldnn INTERFACE)
add_dependencies(libmkldnn ext_mkldnn)
target_include_directories(libmkldnn SYSTEM INTERFACE ${EXTERNAL_PROJECTS_ROOT}/mkldnn/include)
if (WIN32)
    target_link_libraries(libmkldnn INTERFACE
        ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLDNN_IMPLIB}
        libmkl
    )
else()
    target_link_libraries(libmkldnn INTERFACE
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
        libmkl
    )
endif()

if(WIN32)
    install(
        FILES
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
            ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLML_IMPLIB}
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${OMP_LIB}
            ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${OMP_IMPLIB}
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
            ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLDNN_IMPLIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
else()
    install(
        FILES
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${OMP_LIB}
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
endif()
