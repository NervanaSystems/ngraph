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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download MLSL
#------------------------------------------------------------------------------

SET(MLSL_GIT_URL https://github.com/intel/MLSL)
SET(MLSL_GIT_TAG master)

find_program(MAKE_EXE NAMES gmake nmake make)

ExternalProject_Add(
    MLSL
    PREFIX MLSL
    GIT_REPOSITORY ${MLSL_GIT_URL}
    GIT_TAG ${MLSL_GIT_TAG}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE_EXE} -j 1
    INSTALL_COMMAND ${MAKE_EXE} install PREFIX=${EXTERNAL_PROJECTS_ROOT}/MLSL/install
    BUILD_IN_SOURCE TRUE
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/stamp"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/src"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/MLSL/install"
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(MLSL SOURCE_DIR)
ExternalProject_Get_Property(MLSL INSTALL_DIR)
add_library(libmlsl INTERFACE)
target_include_directories(libmlsl SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_link_libraries(libmlsl INTERFACE ${INSTALL_DIR}/intel64/lib/libmlsl.so)
add_dependencies(libmlsl MLSL)

install(DIRECTORY ${INSTALL_DIR}/intel64/lib/
        DESTINATION ${NGRAPH_INSTALL_LIB}
        FILES_MATCHING REGEX "lib*")
install(FILES ${SOURCE_DIR}/include/mlsl.hpp DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph")
