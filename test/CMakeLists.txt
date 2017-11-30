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
    autodiff.cpp
    build_graph.cpp
    builder.cpp
    builder_autobroadcast.cpp
    copy.cpp
    eigen.cpp
    element_type.cpp
    file_util.cpp
    input_output_assign.cpp
    main.cpp
    op.cpp
    pass_liveness.cpp
    pass_manager.cpp
    pass_memory_layout.cpp
    serialize.cpp
    pattern.cpp
    shape.cpp
    tensor.cpp
    topological_sort.cpp
    type_prop.cpp
    util/autodiff/backprop_function.cpp
    util/test_tools.cpp
    util.cpp
    uuid.cpp
)

#================================================================================================
# To auto generate a suite of unit tests for a backend add a line like this
# set(BACKEND_NAMES ${BACKEND_NAMES} "BACKEND_NAME_GOES_HERE")
# and replace BACKEND_NAME_GOES_HERE with your backend name.
# The code for the unit test suite is in test/backend_test.in.cpp
#================================================================================================
set(BACKEND_NAMES ${BACKEND_NAMES} "NGVM")

if(MKLDNN_INCLUDE_DIR)
    include_directories(SYSTEM ${MKLDNN_INCLUDE_DIR})
    link_directories(${MKLDNN_LIB_DIR})
    set(SRC ${SRC} mkldnn.cpp)
endif()

if(NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR)
    include_directories(SYSTEM ${LLVM_INCLUDE_DIR})
    link_directories(${LLVM_LIB_DIR})
    set(SRC ${SRC} codegen.cpp)
    set(BACKEND_NAMES ${BACKEND_NAMES} "CPU")
endif()

if(NGRAPH_GPU_ENABLE)
    set(SRC
        main.cpp
        cudnn.cpp)
    # Disabled for testing
    # set(BACKEND_NAMES ${BACKEND_NAMES} "GPU")
endif()

foreach(BACKEND_NAME ${BACKEND_NAMES})
    configure_file(backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/backend_test_${BACKEND_NAME}.cpp)
    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

add_executable(unit-test ${SRC})

if(MKLDNN_INCLUDE_DIR)
    target_link_libraries(unit-test mkldnn)
    add_dependencies(unit-test ext_mkldnn)
endif()

if(LLVM_INCLUDE_DIR)
    target_link_libraries(unit-test ${LLVM_LINK_LIBS})
    add_dependencies(unit-test ext_llvm)
endif()

if(CUDA_INCLUDE_DIRS)
  target_link_libraries(unit-test ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})
endif()

target_link_libraries(unit-test ngraph libgtest pthread)
target_link_libraries(unit-test ${CMAKE_DL_LIBS})

add_dependencies(unit-test ngraph libgtest eigen)

add_custom_target(style-check
        COMMAND ${PROJECT_SOURCE_DIR}/maint/check-code-format.sh
        )

add_custom_target(unit-test-check
	COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
	DEPENDS unit-test
        )

add_custom_target(check
        DEPENDS
        style-check
        unit-test-check
        )

