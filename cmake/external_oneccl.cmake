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
set(ONECCL_GIT_TAG d2b9499ace634e230ed7b30ebd9e47a7555a8cee)
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
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_INSTALL_PREFIX=${ONECCL_INSTALL_PREFIX}
    INSTALL_DIR "${ONECCL_INSTALL_PREFIX}"
    EXCLUDE_FROM_ALL TRUE
    )

add_library(libccl INTERFACE)
ExternalProject_Get_Property(ext_oneccl SOURCE_DIR INSTALL_DIR)
set(ONECCL_LIB_DIR ${INSTALL_DIR}/lib)
set(ONECCL_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}ccl${CMAKE_SHARED_LIBRARY_SUFFIX})
ExternalProject_Add_Step(
    ext_oneccl
    CopyONECCL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ONECCL_LIB_DIR} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copy oneCCL runtime libraries to ngraph build directory."
    DEPENDEES install
    )
target_include_directories(libccl SYSTEM INTERFACE ${SOURCE_DIR}/include)

set(ONECCL_LINK_LIBRARIES
    ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${ONECCL_LIB})

target_link_libraries(libccl PRIVATE INTERFACE ${ONECCL_LINK_LIBRARIES})
add_dependencies(libccl ext_oneccl)

# installation
# oneccl & mpi & pmi & fabric libraries
install(DIRECTORY "${INSTALL_DIR}/lib"
        DESTINATION ${CMAKE_INSTALL_PREFIX})

#install mpi binaries
install(DIRECTORY "${INSTALL_DIR}/bin/"
        USE_SOURCE_PERMISSIONS
        DESTINATION ${NGRAPH_INSTALL_BIN})

#install mpi tunning data
install(DIRECTORY "${INSTALL_DIR}/etc/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/etc)

#install env scripts
install(DIRECTORY "${INSTALL_DIR}/env"
        DESTINATION ${CMAKE_INSTALL_PREFIX})

#install headers
install(DIRECTORY "${INSTALL_DIR}/include/"
        DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph)
