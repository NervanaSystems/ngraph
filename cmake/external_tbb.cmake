# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

#----------------------------------------------------------------------------------------------------------
# Fetch and install TBB
#----------------------------------------------------------------------------------------------------------

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    set(TBB_GIT_REPO_URL https://github.com/01org/tbb)
    set(TBB_GIT_TAG "tbb_2018")
    set(TBB_INSTALL_DIR ${EXTERNAL_INSTALL_DIR}/tbb)

    # The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
    if(${CMAKE_VERSION} VERSION_LESS 3.2)
        ExternalProject_Add(
            ext_tbb
            GIT_REPOSITORY ${TBB_GIT_REPO_URL}
            GIT_TAG ${TBB_GIT_TAG}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            UPDATE_COMMAND ""
            INSTALL_COMMAND ""
            )
    else()
        ExternalProject_Add(
            ext_tbb
            GIT_REPOSITORY ${TBB_GIT_REPO_URL}
            GIT_TAG ${TBB_GIT_TAG}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            UPDATE_COMMAND ""
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${TBB_INSTALL_DIR}/include/tbb/tbb.h"
            )
    endif()

    ExternalProject_Get_Property(ext_tbb source_dir binary_dir)

    include(${source_dir}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${source_dir} MAKE_ARGS compiler=clang CONFIG_DIR TBB_DIR)
    find_package(TBB REQUIRED tbb)

endif()
