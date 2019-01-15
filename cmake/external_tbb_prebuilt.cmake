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

if (WIN32)
    set(ARCHIVE_FILE_BASE tbb2019_20181203oss)
    set(TBB_FILE https://github.com/01org/tbb/releases/download/2019_U3/${ARCHIVE_FILE_BASE}_win.zip)
    set(TBB_SHA1_HASH 1989458a49e780d76248edac13b963f80c9a460c)
endif()

ExternalProject_Add(
    ext_tbb
    URL ${TBB_FILE}
    URL_HASH SHA1=${TBB_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_tbb SOURCE_DIR)
set(SOURCE_DIR ${SOURCE_DIR}/${ARCHIVE_FILE_BASE})

if (WIN32)
    set(TBB_LINK_LIBS ${SOURCE_DIR}/lib/intel64/vc14/tbb.lib)
else()
    set(TBB_LINK_LIBS
        ${SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}clangTooling${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}clangTooling${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}clangTooling${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
endif()

add_library(libtbb INTERFACE)
add_dependencies(libtbb ext_tbb)
target_include_directories(libtbb SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_link_libraries(libtbb INTERFACE ${TBB_LINK_LIBS})
