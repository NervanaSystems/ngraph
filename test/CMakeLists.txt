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

include_directories(
    SYSTEM
    "${GTEST_INCLUDE_DIR}"
    )

include_directories(
    "${NGRAPH_INCLUDE_DIR}"
    )

set (SRC
    main.cpp
    build_graph.cpp
    util.cpp
    tensor.cpp
    element_type.cpp
    uuid.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

add_executable(unit-test ${SRC})
target_link_libraries(unit-test ngraph pthread libgtest)
add_dependencies(unit-test ngraph libgtest)

add_custom_target(check
	COMMAND ${PROJECT_BINARY_DIR}/test/unit-test
	DEPENDS unit-test)

