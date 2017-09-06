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

# Enable ExternalProject CMake module
include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

#----------------------------------------------------------------------------------------------------------
# Download and install GoogleTest ...
#----------------------------------------------------------------------------------------------------------

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
ExternalProject_Add(
    eigen
    URL http://bitbucket.org/eigen/eigen/get/3.3.3.zip
    # PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    BUILD_BYPRODUCTS "${EXTERNAL_INSTALL_LOCATION}/include/eigen3"
    )

#----------------------------------------------------------------------------------------------------------

ExternalProject_Get_Property(eigen source_dir binary_dir)

set(EIGEN_INCLUDE_DIR "${EXTERNAL_INSTALL_LOCATION}/include/eigen3" PARENT_SCOPE)
