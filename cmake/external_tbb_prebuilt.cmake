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


set(NGRAPH_TBB_VERSION "2019_U3")
set(NGRAPH_TBB_SUB_VERSION "tbb2019_20181203oss")

if (WIN32)
    set(TBB_FILE https://github.com/01org/tbb/releases/download/${NGRAPH_TBB_VERSION}/${NGRAPH_TBB_SUB_VERSION}_win.zip)
    set(TBB_SHA1_HASH 1989458a49e780d76248edac13b963f80c9a460c)
elseif(APPLE)
    set(TBB_FILE https://github.com/01org/tbb/releases/download/${NGRAPH_TBB_VERSION}/${NGRAPH_TBB_SUB_VERSION}_mac.tgz)
    set(TBB_SHA1_HASH 36926fb46add578b88a5c7e19652b94bb612e4be)
endif()

ExternalProject_Add(
    ext_tbb
    PREFIX tbb
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
set(INSTALL_DIR ${SOURCE_DIR}/${NGRAPH_TBB_SUB_VERSION})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_LIB_NAME tbb_debug)
else()
    set(TBB_LIB_NAME tbb)
endif()

if (WIN32)
    set(TBB_LINK_LIBS ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${TBB_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

    ExternalProject_Add_Step(
        ext_tbb
        CopyTBB
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INSTALL_DIR}/bin/intel64/vc14/${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
        COMMENT "Move tbb shared libraries to ngraph build directory"
        DEPENDEES download
        )

    ExternalProject_Add_Step(
        ext_tbb
        CopyTBBIMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INSTALL_DIR}/lib/intel64/vc14/${TBB_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX} ${NGRAPH_ARCHIVE_OUTPUT_DIRECTORY}/${TBB_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        COMMENT "Move tbb libraries to ngraph build directory"
        DEPENDEES download
        )

    install(FILES ${NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY}/${TBB_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
                  ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
            DESTINATION ${NGRAPH_INSTALL_LIB})
elseif(APPLE)
    set(TBB_LINK_LIBS
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
    )

    add_custom_command(TARGET ext_tbb POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
        COMMENT "Move tbb libraries to ngraph build directory"
    )

    install(FILES ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
            DESTINATION ${NGRAPH_INSTALL_LIB})
endif()


add_library(libtbb INTERFACE)
add_dependencies(libtbb ext_tbb)
target_include_directories(libtbb SYSTEM INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(libtbb INTERFACE ${TBB_LINK_LIBS})
