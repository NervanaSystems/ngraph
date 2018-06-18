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

#------------------------------------------------------------------------------
# Fetch and configure TBB
#------------------------------------------------------------------------------

if(NGRAPH_TBB_ENABLE)
    set(TBB_GIT_REPO_URL https://github.com/01org/tbb)
    set(TBB_GIT_TAG "tbb_2018")

    configure_file(${CMAKE_SOURCE_DIR}/cmake/tbb_fetch.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tbb/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")

    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tbb/tbb-src)
endif()
