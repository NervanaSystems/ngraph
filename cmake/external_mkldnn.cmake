# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
set(NGRAPH_MKLDNN_SHORT_VERSION 0)
set(NGRAPH_MKLDNN_FULL_VERSION 0.21)
set(NGRAPH_MKLDNN_ASSET_VERSION "v0.21")
set(NGRAPH_MKLDNN_VERSION "v0.21")
set(NGRAPH_MKLDNN_MKLML_VERSION "2019.0.5.20190502")
set(NGRAPH_MKLDNN_MKLML_WIN32_VERSION "2020.0.20190813") 
set(NGRAPH_MKLDNN_GIT_TAG "v0.21")

#------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#------------------------------------------------------------------------------

set(MKLDNN_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX})
if (LINUX)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLDNN_SHORT_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}.${NGRAPH_MKLDNN_SHORT_VERSION})
    set(MKLDNN_FULL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}.${NGRAPH_MKLDNN_FULL_VERSION})
elseif (APPLE)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLDNN_SHORT_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn.${NGRAPH_MKLDNN_SHORT_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLDNN_FULL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn.${NGRAPH_MKLDNN_FULL_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif (WIN32)
    set(MKLDNN_IMPLIB mkldnn${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(MKLML_LIB mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLML_IMPLIB mklml${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(OMP_LIB libiomp5md${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_IMPLIB libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
    if(NOT LINUX AND NOT WIN32)
        message(FATAL_ERROR "Unsupported platform for prebuilt mkl-dnn!")
    endif()
    if(NOT MKLML_LIB_DIR)
        set(MKLML_LIB_DIR ${MKLDNN_LIB_DIR})
    endif()

    if(WIN32)
        add_library(libmkl STATIC IMPORTED)
        set_property(TARGET libmkl PROPERTY IMPORTED_LOCATION ${MKLML_LIB_DIR}/${MKLML_IMPLIB})
        set_target_properties(libmkl PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES ${MKLML_LIB_DIR}/${OMP_IMPLIB})
    else()
        add_library(libmkl SHARED IMPORTED)
        set_property(TARGET libmkl PROPERTY IMPORTED_LOCATION ${MKLML_LIB_DIR}/${MKLML_LIB})
        set_target_properties(libmkl PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES ${MKLML_LIB_DIR}/${OMP_LIB})
        if(LINUX)
            set_property(TARGET libmkl PROPERTY IMPORTED_NO_SONAME 1)
        endif()
    endif()

    if(WIN32)
        add_library(libmkldnn STATIC IMPORTED)
        set_property(TARGET libmkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_IMPLIB})
        set_target_properties(libmkldnn PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES "${MKLML_LIB_DIR}/${MKLML_IMPLIB};${MKLML_LIB_DIR}/${OMP_IMPLIB}")
    else()
        add_library(libmkldnn SHARED IMPORTED)
        set_property(TARGET libmkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_LIB})
        set_target_properties(libmkldnn PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES "${MKLML_LIB_DIR}/${MKLML_LIB};${MKLML_LIB_DIR}/${OMP_LIB}")
    endif()
    set_target_properties(libmkldnn PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${MKLDNN_INCLUDE_DIR})

    install(FILES ${MKLDNN_LIB_DIR}/${MKLDNN_LIB} ${MKLML_LIB_DIR}/${MKLML_LIB} ${MKLML_LIB_DIR}/${OMP_LIB}  DESTINATION ${NGRAPH_INSTALL_LIB})
    return()
endif()

# This section sets up MKL as an external project to be used later by MKLDNN

set(MKLURLROOT "https://github.com/intel/mkl-dnn/releases/download/${NGRAPH_MKLDNN_ASSET_VERSION}/")
set(MKLVERSION ${NGRAPH_MKLDNN_MKLML_VERSION})
set(MKLWIN32VERSION ${NGRAPH_MKLDNN_MKLML_WIN32_VERSION})
if (LINUX)
    set(MKLPACKAGE "mklml_lnx_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 6ab490f0b358124338d04ee9383c3cbc536969d8)
elseif (APPLE)
    set(MKLPACKAGE "mklml_mac_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH a1c42af04f990b0e515a1c31946424b2e68fccc9)
elseif (WIN32)
    set(MKLPACKAGE "mklml_win_${MKLWIN32VERSION}.zip")
    set(MKL_SHA1_HASH cc117093e658d50a8e4e3d1cf192c300b6bac0fc)
endif()
set(MKL_LIBS ${MKLML_LIB} ${OMP_LIB})
set(MKLURL ${MKLURLROOT}${MKLPACKAGE})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(APPLE)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 8.0)
            set(MKLDNN_FLAG "-Wno-stringop-truncation -Wno-stringop-overflow")
        endif()
    elseif(LINUX)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.8.2)
            #pragma GCC diagnostic ignored does not work on GCC used for manylinux1
            set(MKLDNN_FLAG "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds")
            set(MKLDNN_FLAG "${MKLDNN_FLAG} -Wno-unused-result -Wno-unused-value")
        endif()
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
    BUILD_BYPRODUCTS ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
                     ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${OMP_LIB}
)

set(MKLDNN_DEPENDS ext_mkl)
ExternalProject_Get_Property(ext_mkl source_dir)
set(MKL_ROOT ${EXTERNAL_PROJECTS_ROOT}/mkldnn/src/external/mkl)
set(MKL_SOURCE_DIR ${source_dir})

ExternalProject_Add_Step(
    ext_mkl
    CopyMKL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${MKLML_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
    COMMENT "Copy mklml runtime libraries to ngraph build directory."
    DEPENDEES download
    )

ExternalProject_Add_Step(
    ext_mkl
    CopyOMP
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${OMP_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${OMP_LIB}
    COMMENT "Copy OpenMP runtime libraries to ngraph build directory."
    DEPENDEES download
    )

if(WIN32)
    ExternalProject_Add_Step(
        ext_mkl
        CopyMKLIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${MKLML_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLML_IMPLIB}
        COMMENT "Copy mklml runtime libraries to ngraph build directory."
        DEPENDEES download
        )

    ExternalProject_Add_Step(
        ext_mkl
        CopyOMPIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKL_SOURCE_DIR}/lib/${OMP_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${OMP_IMPLIB}
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

# Revert prior changes to make incremental build work.
set(MKLDNN_PATCH_REVERT_COMMAND cd ${EXTERNAL_PROJECTS_ROOT}/mkldnn/src && git reset HEAD --hard)

if (WIN32)
    ExternalProject_Add(
        ext_mkldnn
        PREFIX mkldnn
        DEPENDS ${MKLDNN_DEPENDS}
        GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
        GIT_TAG ${MKLDNN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        PATCH_COMMAND ${MKLDNN_PATCH_REVERT_COMMAND}
        COMMAND git apply --ignore-space-change --ignore-whitespace ${CMAKE_SOURCE_DIR}/cmake/${MKLDNN_PATCH_FILE}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DMKLDNN_BUILD_TESTS=FALSE
            -DMKLDNN_BUILD_EXAMPLES=FALSE
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
    if(LINUX)
        set(MKLDNN_RPATH "-DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}")
    endif()
    ExternalProject_Add(
        ext_mkldnn
        PREFIX mkldnn
        DEPENDS ${MKLDNN_DEPENDS}
        GIT_REPOSITORY ${MKLDNN_GIT_REPO_URL}
        GIT_TAG ${MKLDNN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        PATCH_COMMAND ${MKLDNN_PATCH_REVERT_COMMAND}
        COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/${MKLDNN_PATCH_FILE}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DWITH_TEST=FALSE
            -DWITH_EXAMPLE=FALSE
            -DMKLDNN_BUILD_TESTS=FALSE
            -DMKLDNN_BUILD_EXAMPLES=FALSE
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/mkldnn
            ${MKLDNN_RPATH}
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
        BUILD_BYPRODUCTS ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
        )
endif()

if(WIN32)
    ExternalProject_Add_Step(
        ext_mkldnn
        CopyMKLDNN
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/bin/${MKLDNN_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
        COMMENT "Copy mkldnn runtime libraries to ngraph build directory."
        DEPENDEES install
        )

    ExternalProject_Add_Step(
        ext_mkldnn
        CopyMKLDNNIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/${MKLDNN_IMPLIB} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${MKLDNN_IMPLIB}
        COMMENT "Copy mkldnn runtime libraries to ngraph build directory."
        DEPENDEES install
        )
else()
    ExternalProject_Add_Step(
        ext_mkldnn
        CopyMKLDNN
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/${CMAKE_INSTALL_LIBDIR}/${MKLDNN_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_LIB}
        COMMENT "Copy mkldnn runtime libraries to ngraph build directory."
        DEPENDEES install
        )
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        ExternalProject_Add_Step(
            ext_mkldnn
            CopyMKLDNNEXTRA
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/${CMAKE_INSTALL_LIBDIR}/${MKLDNN_SHORT_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_SHORT_LIB}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EXTERNAL_PROJECTS_ROOT}/mkldnn/${CMAKE_INSTALL_LIBDIR}/${MKLDNN_FULL_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLDNN_FULL_LIB}
            COMMENT "Copy extra mkldnn runtime libraries to ngraph build directory."
            DEPENDEES install
           )
    endif()
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
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLML_LIB}
            ${NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY}/${MKLML_IMPLIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${OMP_LIB}
            ${NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY}/${OMP_IMPLIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLDNN_LIB}
            ${NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY}/${MKLDNN_IMPLIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
else()
    install(
        FILES
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLML_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${OMP_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLDNN_LIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        install(
            FILES
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLDNN_SHORT_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLDNN_FULL_LIB}
            DESTINATION
                ${NGRAPH_INSTALL_LIB}
            OPTIONAL
            )
    endif()
endif()
