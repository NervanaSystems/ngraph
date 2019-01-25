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

# # Enable ExternalProject CMake module
# include(ExternalProject)

# #------------------------------------------------------------------------------
# # Download and install OpenMPI.
# #------------------------------------------------------------------------------

# set(OMPI_GIT_REPO_URL https://github.com/open-mpi/ompi.git)
# set(OMPI_GIT_BRANCH v2.1.1)

# # The 'BUILD_BYPRODUCTS' arguments was introduced in CMake 3.2.

# # ExternalProject_Add(
# # 	    ext_ompi
# # 	        PREFIX ext_ompi
# # 		    GIT_REPOSITORY ${OMPI_GIT_REPO_URL}
# # 			GIT_TAG ${OMPI_GIT_BRANCH}
# # 			INSTALL_COMMAND ""
# # 			UPDATE_COMMAND ""
# # 			PATCH_COMMAND ""
# # 			CONFIGURE_COMMAND ./autogen.pl && ./configure --prefix=${EXTERNAL_PROJECTS_ROOT}/ompi/install
# # 			TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi/tmp"
# # 			STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi/stamp"
# # 			DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi/download"
# # 			SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi/src"
# # 			BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi/src"
# # 			INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/ompi"
# # 			EXCLUDE_FROM_ALL TRUE
# # 			)

# # 			ExternalProject_Get_Property(ext_ompi SOURCE_DIR BINARY_DIR)
# # 			ExternalProject_Get_Property(ext_ompi INSTALL_DIR)
# # 			add_library(libompi INTERFACE)
# # 			target_include_directories(libompi SYSTEM INTERFACE ${SOURCE_DIR}/include)
# # 			# target_link_libraries(libompi INTERFACE ${INSTALL_DIR}/lib/libompi.so)
# # 			add_dependencies(libompi ext_ompi)

include(ExternalProject)

find_package(ZLIB REQUIRED)

set(OMPI_TARBALL_URL https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.1.tar.gz)

set(OMPI_SHA1_HASH cbbae7608fe67d3bc79145b03e28af7b69f12800)


ExternalProject_Add(
    ext_ompi
    URL ${OMPI_TARBALL_URL}
    URL_HASH SHA1=${OMPI_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_ompi SOURCE_DIR)

set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} tinfo z m)


