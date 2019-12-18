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

# Add files here
target_sources (ngraph PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/layout_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/layout_converter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/matmul.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/pool.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/reduce_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/reduce_sum.hpp
    )

if (NGRAPH_FLUID_TEST_ENABLE)
    add_subdirectory(operators/test)
endif()
