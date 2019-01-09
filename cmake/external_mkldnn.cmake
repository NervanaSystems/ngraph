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
    set(MKLML_LIB mklml${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OMP_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}iomp5md${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if(MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
    if(NOT LINUX)
        message(FATAL_ERROR "Unsupported platform for prebuilt mkl-dnn!")
    endif()
    if(NOT MKLML_LIB_DIR)
        set(MKLML_LIB_DIR ${MKLDNN_LIB_DIR})
    endif()
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
        ${MKLDNN_LIB_DIR}/${MKLDNN_LIB}
        ${MKLDNN_LIB_DIR}/${MKLML_LIB}
        ${MKLDNN_LIB_DIR}/${OMP_LIB}
        )

    install(FILES ${MKLDNN_LIB_DIR}/libmkldnn.so ${MKLML_LIB_DIR}/libmklml_intel.so ${MKLML_LIB_DIR}/libiomp5.so  DESTINATION ${NGRAPH_INSTALL_LIB})
    return()
endif()

# This section sets up MKL as an external project to be used later by MKLDNN

set(MKLURLROOT "https://github.com/intel/mkl-dnn/releases/download/v0.17/")
set(MKLVERSION "2019.0.1.20180928")

if (LINUX)
    set(MKLPACKAGE "mklml_lnx_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 0d9cc8bfc2c1a1e3df5e0b07e2f363bbf934a7e9)
elseif (APPLE)
    set(MKLPACKAGE "mklml_mac_${MKLVERSION}.tgz")
    set(MKL_SHA1_HASH 232787f41cb42f53f5b7e278d8240f6727896133)
elseif (WIN32)
    set(MKLPACKAGE "mklml_win_${MKLVERSION}.zip")
    set(MKL_SHA1_HASH 97f01ab854d8ee88cc0429f301df84844d7cce6b)
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
set(MKLDNN_CFLAG "${CMAKE_C_FLAGS} ${MKLDNN_FLAG}")
set(MKLDNN_CXXFLAG "${CMAKE_CXX_FLAGS} ${MKLDNN_FLAG}")
set(MKLDNN_DEPENDS ext_mkl)

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
    list(APPEND MKL_LIBS_PATH ${MKL_SOURCE_DIR}/lib/${LIB})
endforeach()
target_link_libraries(libmkl INTERFACE ${MKL_LIBS_PATH})

set(MKLDNN_GIT_REPO_URL https://github.com/intel/mkl-dnn)
set(MKLDNN_GIT_TAG "830a100")
if(NGRAPH_LIB_VERSIONING_ENABLE)
    set(MKLDNN_PATCH_FILE mkldnn.patch)
else()
    set(MKLDNN_PATCH_FILE mkldnn_no_so_link.patch)
endif()
set(MKLDNN_LIBS ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/${MKLDNN_LIB})

ExternalProject_Add(
    ext_mkldnn
    DEPENDS ${MKLDNN_DEPENDS}
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
        "-DARCH_OPT_FLAGS=-march=${NGRAPH_TARGET_ARCH} -mtune=${NGRAPH_TARGET_ARCH}"
        -DCMAKE_C_FLAGS=${MKLDNN_CFLAG}
        -DCMAKE_CXX_FLAGS=${MKLDNN_CXXFLAG}
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn/build"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/mkldnn"
    EXCLUDE_FROM_ALL TRUE
    )


# CPU backend has dependency on CBLAS
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
    ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/${MKLDNN_LIB}
    libmkl
    )

install(DIRECTORY ${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/ DESTINATION ${NGRAPH_INSTALL_LIB} OPTIONAL)
