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

# Includes blas 3.8.0 in dnnl
set(NGRAPH_DNNL_SHORT_VERSION 1)
set(NGRAPH_DNNL_FULL_VERSION 1.5)
set(NGRAPH_DNNL_MKLML_ASSET_VERSION "v0.21")
set(NGRAPH_DNNL_MKLML_VERSION "2019.0.5.20190502")
set(NGRAPH_DNNL_MKLML_WIN32_VERSION "2020.0.20190813")
set(NGRAPH_DNNL_GIT_TAG "v1.5.1")

#------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#------------------------------------------------------------------------------

set(DNNL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}dnnl${CMAKE_SHARED_LIBRARY_SUFFIX})
if (LINUX)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DNNL_SHORT_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}dnnl${CMAKE_SHARED_LIBRARY_SUFFIX}.${NGRAPH_DNNL_SHORT_VERSION})
    set(DNNL_FULL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}dnnl${CMAKE_SHARED_LIBRARY_SUFFIX}.${NGRAPH_DNNL_FULL_VERSION})
elseif (APPLE)
    set(MKLML_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DNNL_SHORT_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}dnnl.${NGRAPH_DNNL_SHORT_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DNNL_FULL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}dnnl.${NGRAPH_DNNL_FULL_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif (WIN32)
    set(DNNL_IMPLIB dnnl${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(MKLML_LIB mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLML_IMPLIB mklml${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(OMP_LIB libiomp5md${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_IMPLIB libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(DNNL_INCLUDE_DIR AND DNNL_LIB_DIR)
    if(NOT LINUX AND NOT WIN32)
        message(FATAL_ERROR "Unsupported platform for prebuilt mkl-dnn!")
    endif()
    if(NOT MKLML_LIB_DIR)
        set(MKLML_LIB_DIR ${DNNL_LIB_DIR})
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
        add_library(libdnnl STATIC IMPORTED)
        set_property(TARGET libdnnl PROPERTY IMPORTED_LOCATION ${DNNL_LIB_DIR}/${DNNL_IMPLIB})
        set_target_properties(libdnnl PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES "${MKLML_LIB_DIR}/${MKLML_IMPLIB};${MKLML_LIB_DIR}/${OMP_IMPLIB}")
    else()
        add_library(libdnnl SHARED IMPORTED)
        set_property(TARGET libdnnl PROPERTY IMPORTED_LOCATION ${DNNL_LIB_DIR}/${DNNL_LIB})
        set_target_properties(libdnnl PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES "${MKLML_LIB_DIR}/${MKLML_LIB};${MKLML_LIB_DIR}/${OMP_LIB}")
    endif()
    set_target_properties(libdnnl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR})

    install(FILES ${DNNL_LIB_DIR}/${DNNL_LIB} ${MKLML_LIB_DIR}/${MKLML_LIB} ${MKLML_LIB_DIR}/${OMP_LIB}  DESTINATION ${NGRAPH_INSTALL_LIB})
    add_library(DNNL::dnnl ALIAS libdnnl)
    return()
endif()

# This section sets up MKL as an external project to be used later by DNNL

set(MKLURLROOT "https://github.com/oneapi-src/oneDNN/releases/download/${NGRAPH_DNNL_MKLML_ASSET_VERSION}/")
set(MKLVERSION ${NGRAPH_DNNL_MKLML_VERSION})
set(MKLWIN32VERSION ${NGRAPH_DNNL_MKLML_WIN32_VERSION})
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
            set(DNNL_FLAG "-Wno-stringop-truncation -Wno-stringop-overflow")
        endif()
    elseif(LINUX)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.8.2)
            #pragma GCC diagnostic ignored does not work on GCC used for manylinux1
            set(DNNL_FLAG "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds")
            set(DNNL_FLAG "${DNNL_FLAG} -Wno-unused-result -Wno-unused-value")
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

set(DNNL_DEPENDS ext_mkl)
ExternalProject_Get_Property(ext_mkl source_dir)
set(MKL_ROOT ${EXTERNAL_PROJECTS_ROOT}/dnnl/src/external/mkl)
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

set(DNNL_GIT_REPO_URL https://github.com/oneapi-src/oneDNN)
set(DNNL_GIT_TAG ${NGRAPH_DNNL_GIT_TAG})
set(DNNL_PATCH_FILE onednn.patch)
set(DNNL_LIBS ${EXTERNAL_PROJECTS_ROOT}/dnnl/lib/${DNNL_LIB})

# Revert prior changes to make incremental build work.
set(DNNL_PATCH_REVERT_COMMAND cd ${EXTERNAL_PROJECTS_ROOT}/dnnl/src && git reset HEAD --hard)

if (WIN32)
    ExternalProject_Add(
        ext_dnnl
        PREFIX dnnl
        DEPENDS ${DNNL_DEPENDS}
        GIT_REPOSITORY ${DNNL_GIT_REPO_URL}
        GIT_TAG ${DNNL_GIT_TAG}
        GIT_SHALLOW 1
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        PATCH_COMMAND ${DNNL_PATCH_REVERT_COMMAND}
        COMMAND git apply --ignore-space-change --ignore-whitespace ${CMAKE_SOURCE_DIR}/cmake/${DNNL_PATCH_FILE}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DDNNL_BUILD_TESTS=FALSE
            -DDNNL_BUILD_EXAMPLES=FALSE
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/dnnl
            -DDNNL_ENABLE_CONCURRENT_EXEC=ON
            -DDNNL_CPU_RUNTIME=${NGRAPH_CPU_RUNTIME}
            -DDNNL_LIB_VERSIONING_ENABLE=${NGRAPH_LIB_VERSIONING_ENABLE}
            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl"
        EXCLUDE_FROM_ALL TRUE
        )
else()
    if(LINUX)
        set(DNNL_RPATH "-DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}")
    endif()
    ExternalProject_Add(
        ext_dnnl
        PREFIX dnnl
        DEPENDS ${DNNL_DEPENDS}
        GIT_REPOSITORY ${DNNL_GIT_REPO_URL}
        GIT_TAG ${DNNL_GIT_TAG}
        GIT_SHALLOW 1
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        PATCH_COMMAND ${DNNL_PATCH_REVERT_COMMAND}
        COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/${DNNL_PATCH_FILE}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DDNNL_BUILD_TESTS=FALSE
            -DDNNL_BUILD_EXAMPLES=FALSE
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_ROOT}/dnnl
            ${DNNL_RPATH}
            -DDNNL_ENABLE_CONCURRENT_EXEC=ON
            -DDNNL_CPU_RUNTIME=${NGRAPH_CPU_RUNTIME}
            -DDNNL_LIB_VERSIONING_ENABLE=${NGRAPH_LIB_VERSIONING_ENABLE}
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
            "-DDNNL_ARCH_OPT_FLAGS=-march=${NGRAPH_TARGET_ARCH} -mtune=${NGRAPH_TARGET_ARCH} ${DNNL_FLAG}"
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/dnnl"
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${DNNL_LIB}
        )
endif()

ExternalProject_Add_Step(
    ext_dnnl
    PrepareMKL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MKL_SOURCE_DIR} ${MKL_ROOT}
    DEPENDEES download
    DEPENDERS configure
    )

add_library(libdnnl INTERFACE)
add_dependencies(libdnnl ext_dnnl)
target_include_directories(libdnnl SYSTEM INTERFACE ${EXTERNAL_PROJECTS_ROOT}/dnnl/include)
if (WIN32)
    target_link_libraries(libdnnl INTERFACE
        ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${DNNL_IMPLIB}
        libmkl
    )
else()
    target_link_libraries(libdnnl INTERFACE
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${DNNL_LIB}
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
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${DNNL_LIB}
            ${NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY}/${DNNL_IMPLIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
else()
    install(
        FILES
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${MKLML_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${OMP_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${DNNL_LIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        install(
            FILES
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${DNNL_SHORT_LIB}
            ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${DNNL_FULL_LIB}
            DESTINATION
                ${NGRAPH_INSTALL_LIB}
            OPTIONAL
            )
    endif()
endif()
add_library(DNNL::dnnl ALIAS libdnnl)
