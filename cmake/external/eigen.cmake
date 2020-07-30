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

if(TARGET Eigen3::Eigen)
    return()
endif()

include(FetchContent)

message(STATUS "Fetching Eigen3")

set(EIGEN_GIT_TAG dcf7655b3d469a399c1182f350c9009e13ad8654)
set(EIGEN_GIT_URL https://gitlab.com/libeigen/eigen.git)

FetchContent_Declare(ext_eigen
    GIT_REPOSITORY ${EIGEN_GIT_URL}
    GIT_TAG ${EIGEN_GIT_TAG}
    )

set(BUILD_TESTING OFF CACHE INTERNAL "")

FetchContent_GetProperties(ext_eigen)
if(NOT ext_eigen_POPULATED)
    FetchContent_Populate(ext_eigen)
endif()

add_library(libeigen INTERFACE)
target_include_directories(libeigen INTERFACE ${ext_eigen_SOURCE_DIR})
# Prevent Eigen from using any LGPL3 code
target_compile_definitions(libeigen INTERFACE EIGEN_MPL2_ONLY)
add_library(Eigen3::Eigen ALIAS libeigen)
