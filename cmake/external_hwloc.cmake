# ******************************************************************************
# Copyright 2019 Intel Corporation
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

if(NOT NGRAPH_MANYLINUX_ENABLE)
    return()
endif()

include(ExternalProject)

set(HWLOC_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/hwloc)

ExternalProject_Add(
    ext_hwloc
    PREFIX hwloc
    URL https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz
    URL_HASH SHA1=39d5a99d14a0810139b10c222f4b6aa4ceb45c70
    DOWNLOAD_NO_PROGRESS TRUE
    CONFIGURE_COMMAND ${HWLOC_INSTALL_PREFIX}/src/ext_hwloc/configure --prefix=${HWLOC_INSTALL_PREFIX} --disable-cairo
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_hwloc INSTALL_DIR)

ExternalProject_Add_Step(
    ext_hwloc
    CopyHWLOC
    COMMAND ${CMAKE_COMMAND} -E copy ${INSTALL_DIR}/lib/libhwloc.so.15 ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/libhwloc.so.15
    COMMENT "Copy hwloc runtime libraries to ngraph build directory."
    DEPENDEES install
    )


install(
    FILES
        ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/libhwloc.so.15
    DESTINATION
        ${NGRAPH_INSTALL_LIB}
    OPTIONAL
    )
