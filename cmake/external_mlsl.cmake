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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download MLSL
#------------------------------------------------------------------------------

set(MLSL_GIT_URL https://github.com/intel/MLSL)
set(MLSL_GIT_TAG 98a683cb861514259480aff2e54c8fce4bec67e5)

find_program(MAKE_EXE NAMES gmake nmake make)

ExternalProject_Add(
    MLSL
    PREFIX MLSL
    GIT_REPOSITORY ${MLSL_GIT_URL}
    GIT_TAG ${MLSL_GIT_TAG}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE_EXE} -j 1 ENABLE_INTERNAL_ENV_UPDATE=1
    INSTALL_COMMAND ${MAKE_EXE} install PREFIX=${EXTERNAL_PROJECTS_ROOT}/MLSL/install
    BUILD_IN_SOURCE TRUE
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/stamp"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/src"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/install"
    EXCLUDE_FROM_ALL TRUE
    )

add_library(libmlsl INTERFACE)
ExternalProject_Get_Property(MLSL SOURCE_DIR)
ExternalProject_Get_Property(MLSL INSTALL_DIR)
set(MLSL_LIB_DIR ${INSTALL_DIR}/intel64/lib/thread)
set(MLSL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mlsl${CMAKE_SHARED_LIBRARY_SUFFIX})
set(MPI_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mpi${CMAKE_SHARED_LIBRARY_SUFFIX})
set(FABRIC_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}fabric${CMAKE_SHARED_LIBRARY_SUFFIX})
ExternalProject_Add_Step(
    MLSL
    CopyMLSL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MLSL_LIB_DIR} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy mlsl runtime libraries to ngraph build directory."
    DEPENDEES install
    )
target_include_directories(libmlsl SYSTEM INTERFACE ${SOURCE_DIR}/include)

set(MLSL_LINK_LIBRARIES
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MLSL_LIB}
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MPI_LIB}
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${FABRIC_LIB})

target_link_libraries(libmlsl PRIVATE INTERFACE ${MLSL_LINK_LIBRARIES})
add_dependencies(libmlsl MLSL)

#installation
#mlsl & mpi & fabric libraries
install(DIRECTORY "${INSTALL_DIR}/intel64/lib/thread/"
        DESTINATION ${NGRAPH_INSTALL_LIB})

#install mpi binaries
install(DIRECTORY "${INSTALL_DIR}/intel64/bin/thread/"
        USE_SOURCE_PERMISSIONS
        DESTINATION ${NGRAPH_INSTALL_BIN})

#install mpi tunning data
install(DIRECTORY "${INSTALL_DIR}/intel64/etc/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/etc)

#mlsl header
install(FILES ${SOURCE_DIR}/include/mlsl.hpp
        DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph)
