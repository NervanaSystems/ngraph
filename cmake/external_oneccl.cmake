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
# Download oneCCL
#------------------------------------------------------------------------------

set(ONECCL_GIT_URL https://github.com/intel/oneccl)
set(ONECCL_GIT_TAG 4ecffc589f09609d860710d5b8ada376f942bc35)
set(ONECCL_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/oneccl/install)

ExternalProject_Add(
    ext_oneccl
    PREFIX oneccl
    GIT_REPOSITORY ${ONECCL_GIT_URL}
    GIT_TAG ${ONECCL_GIT_TAG}
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        -DCMAKE_CXX_FLAGS=${CMAKE_ORIGINAL_CXX_FLAGS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${ONECCL_INSTALL_PREFIX}
    INSTALL_DIR "${ONECCL_INSTALL_PREFIX}"
    EXCLUDE_FROM_ALL TRUE
    )

add_library(libccl INTERFACE)
ExternalProject_Get_Property(ext_oneccl SOURCE_DIR INSTALL_DIR)
set(ONECCL_LIB_DIR ${INSTALL_DIR}/lib)
set(ONECCL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}ccl${CMAKE_SHARED_LIBRARY_SUFFIX})
set(MPI_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}mpi${CMAKE_SHARED_LIBRARY_SUFFIX})
set(PMI_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}pmi${CMAKE_SHARED_LIBRARY_SUFFIX})
set(FABRIC_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}fabric${CMAKE_SHARED_LIBRARY_SUFFIX})
ExternalProject_Add_Step(
    ext_oneccl
    CopyONECCL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ONECCL_LIB_DIR} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy oneCCL runtime libraries to ngraph build directory."
    DEPENDEES install
    )
target_include_directories(libccl SYSTEM INTERFACE ${SOURCE_DIR}/include)

set(ONECCL_LINK_LIBRARIES
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${ONECCL_LIB}
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MPI_LIB}
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${PMI_LIB}
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${FABRIC_LIB})

target_link_libraries(libccl PRIVATE INTERFACE ${ONECCL_LINK_LIBRARIES})
add_dependencies(libccl ext_oneccl)

# installation
# oneccl & mpi & pmi & fabric libraries
install(DIRECTORY "${INSTALL_DIR}/lib/"
        DESTINATION ${NGRAPH_INSTALL_LIB})

#install mpi binaries
install(DIRECTORY "${INSTALL_DIR}/bin/"
        USE_SOURCE_PERMISSIONS
        DESTINATION ${NGRAPH_INSTALL_BIN})

#install mpi tunning data
install(DIRECTORY "${INSTALL_DIR}/etc/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/etc)

#install headers
install(DIRECTORY "${INSTALL_DIR}/include/"
        DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph)
