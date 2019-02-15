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

if(NGRAPH_TBB_ENABLE)
    set(TBB_GIT_REPO_URL https://github.com/01org/tbb)
    set(TBB_GIT_TAG "2019_U2")

    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tbb/tbb-src)

    configure_file(${CMAKE_SOURCE_DIR}/cmake/tbb_fetch.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tbb/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")

    include(${TBB_ROOT}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/tbb_build
                tbb_build_prefix=tbb CONFIG_DIR TBB_DIR)
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
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/${TBB_BUILDDIR_NAME}/
        DESTINATION ${NGRAPH_INSTALL_LIB}
        FILES_MATCHING REGEX "/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}(\\.[0-9]+)*$"
    )
    add_library(libtbb INTERFACE)
    target_link_libraries(libtbb INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/tbb_build/${TBB_BUILDDIR_NAME}/${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    target_include_directories(libtbb SYSTEM INTERFACE ${TBB_ROOT}/include)
endif()
