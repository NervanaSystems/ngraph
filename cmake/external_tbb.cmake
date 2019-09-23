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

#------------------------------------------------------------------------------
# Fetch and configure TBB
#------------------------------------------------------------------------------

set(TBB_GIT_REPO_URL https://github.com/01org/tbb)
set(NGRAPH_TBB_VERSION "2019_U3")

if(NGRAPH_TBB_ENABLE)
    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tbb/tbb-src)

    configure_file(${CMAKE_SOURCE_DIR}/cmake/tbb_fetch.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tbb/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")

    include(${TBB_ROOT}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/tbb_build
        tbb_build_prefix=tbb stdver=c++${NGRAPH_CXX_STANDARD} CONFIG_DIR TBB_DIR)
    find_package(TBB REQUIRED tbb)
    if (NOT TBB_FOUND)
        message(FATAL_ERROR "TBB is needed by the CPU backend and was not found")
    else()
        message(STATUS "Found TBB and imported target ${TBB_IMPORTED_TARGETS}")
    endif()

    set_source_files_properties(cpu_external_function.cpp
        PROPERTIES COMPILE_DEFINITIONS "NGRAPH_TBB_ENABLE")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(TBB_LIB_NAME tbb_debug)
        set(TBB_BUILDDIR_NAME tbb_debug)
    else()
        set(TBB_LIB_NAME tbb)
        set(TBB_BUILDDIR_NAME tbb_release)
    endif()
    set(TBB_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/${TBB_BUILDDIR_NAME})
    set(TBB_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
    file(COPY
             ${TBB_BUILD_DIR}/${TBB_LIB}
         DESTINATION ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY})
    if(LINUX)
        execute_process(COMMAND grep TBB_COMPATIBLE_INTERFACE_VERSION ${TBB_ROOT}/include/tbb/tbb_stddef.h OUTPUT_VARIABLE TBB_SOVER_LINE)
        string(REGEX MATCH "[0-9.]+" TBB_SOVER ${TBB_SOVER_LINE})
        message(STATUS "TBB so version: ${TBB_SOVER}")
        file(COPY
                ${TBB_BUILD_DIR}/${TBB_LIB}.${TBB_SOVER}
             DESTINATION ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY})
    endif()
    install(FILES ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${TBB_LIB}
        DESTINATION ${NGRAPH_INSTALL_LIB})
    if(LINUX)
        install(FILES ${NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY}/${TBB_LIB}.${TBB_SOVER}
            DESTINATION ${NGRAPH_INSTALL_LIB})
    endif()
    add_library(libtbb INTERFACE)
    target_link_libraries(libtbb INTERFACE
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${TBB_LIB}
    )
    target_include_directories(libtbb SYSTEM INTERFACE ${TBB_ROOT}/include)
endif()
