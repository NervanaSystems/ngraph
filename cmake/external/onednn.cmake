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

include(FetchContent)

#------------------------------------------------------------------------------
# Fetch and install MKLML
#------------------------------------------------------------------------------

set(NGRAPH_DNNL_MKLML_ASSET_VERSION "v0.21")
set(NGRAPH_DNNL_MKLML_VERSION "2019.0.5.20190502")
set(NGRAPH_DNNL_MKLML_WIN32_VERSION "2020.0.20190813")
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

message(STATUS "Fetching MKLML")

FetchContent_Declare(
    ext_mkl
    URL ${MKLURL}
    URL_HASH SHA1=${MKL_SHA1_HASH}
)

FetchContent_GetProperties(ext_mkl)
if(NOT ext_mkl_POPULATED)
    FetchContent_Populate(ext_mkl)
endif()

add_library(libmkl INTERFACE)
add_dependencies(libmkl ext_mkl)

if (LINUX)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(MKLML_LIB ${ext_mkl_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}mklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(OMP_LIB ${ext_mkl_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        set(MKLML_LIB ${ext_mkl_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}mklml_gnu${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
elseif (APPLE)
    set(MKLML_LIB ${ext_mkl_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${ext_mkl_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}iomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif (WIN32)
    set(MKLML_LIB ${ext_mkl_SOURCE_DIR}/lib/mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MKLML_IMPLIB ${ext_mkl_SOURCE_DIR}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(OMP_LIB ${ext_mkl_SOURCE_DIR}/lib/libiomp5md${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_IMPLIB ${ext_mkl_SOURCE_DIR}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(WIN32)
    target_link_libraries(libmkl INTERFACE
        ${MKLML_IMPLIB}
        ${OMP_IMPLIB})
else()
    target_link_libraries(libmkl INTERFACE
        ${MKLML_LIB}
        ${OMP_LIB})
endif()

install(
    FILES
        ${MKLML_LIB}
        ${MKLML_IMPLIB}
        ${OMP_LIB}
        ${OMP_IMPLIB}
    DESTINATION
        ${NGRAPH_INSTALL_LIB}
    OPTIONAL
    )

#------------------------------------------------------------------------------
# Fetch and install oneDNN
#------------------------------------------------------------------------------

if(TARGET DNNL::dnnl)
    return()
endif()

if(TARGET dnnl)
    add_library(DNNL::dnnl ALIAS dnnl)
    return()
endif()

set(DNNL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)
if((NOT WIN32) AND NGRAPH_NATIVE_ARCH_ENABLE)
    set(DNNL_ARCH_OPT_FLAGS "-march=${NGRAPH_TARGET_ARCH} -mtune=${NGRAPH_TARGET_ARCH}" CACHE INTERNAL "" FORCE)
endif()
set(DNNL_LIB_VERSIONING_ENABLE ${NGRAPH_LIB_VERSIONING_ENABLE} CACHE INTERNAL "" FORCE)

message(STATUS "Fetching oneDNN")

FetchContent_Declare(
    ext_dnnl
    URL       https://github.com/oneapi-src/oneDNN/archive/v1.6.1.zip
    URL_HASH  SHA1=5ebbe215ac1dd3121fe34511c9ffb597ec1d7a48
)

FetchContent_GetProperties(ext_dnnl)
if(NOT ext_dnnl_POPULATED)
    FetchContent_Populate(ext_dnnl)
    add_subdirectory(${ext_dnnl_SOURCE_DIR} ${ext_dnnl_BINARY_DIR})
    if(NOT NGRAPH_LIB_VERSIONING_ENABLE)
        # Unset VERSION and SOVERSION
        set_property(TARGET dnnl PROPERTY VERSION)
        set_property(TARGET dnnl PROPERTY SOVERSION)
    endif()
endif()

add_library(DNNL::dnnl ALIAS dnnl)
