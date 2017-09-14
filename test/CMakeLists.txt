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
    ${GTEST_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    )

include_directories(
    ${NGRAPH_INCLUDE_DIR}
    )

set (SRC
    build_graph.cpp
    eigen.cpp
    element_type.cpp
    execute.cpp
    input_output_assign.cpp
    main.cpp
    op.cpp
    pass_liveness.cpp
    pass_manager.cpp
    pass_memory_layout.cpp
    runtime.cpp
    shape.cpp
    tensor.cpp
    test_tools.cpp
    topological_sort.cpp
    type_prop.cpp
    util.cpp
    uuid.cpp
)

if(MKLDNN_INCLUDE_DIR)
    include_directories(${MKLDNN_INCLUDE_DIR})
    link_directories(${MKLDNN_LIB_DIR})
    set(SRC ${SRC} mkldnn.cpp)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

add_executable(unit-test ${SRC})

if(MKLDNN_INCLUDE_DIR)
    target_link_libraries(unit-test mkldnn)
    add_dependencies(unit-test ext_mkldnn)
endif()

target_link_libraries(unit-test ngraph libgtest pthread)
target_link_libraries(unit-test ${CMAKE_DL_LIBS})

add_dependencies(unit-test ngraph libgtest eigen)

add_custom_target(check
	COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
	DEPENDS unit-test)
