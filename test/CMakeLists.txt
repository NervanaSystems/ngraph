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

set(SRC
    algebraic_simplification.cpp
    assertion.cpp
    builder_autobroadcast.cpp
    build_graph.cpp
    constant_folding.cpp
    copy.cpp
    core_fusion.cpp
    cpio.cpp
    cse.cpp
    element_type.cpp
    file_util.cpp
    all_close_f.cpp
    inliner.cpp
    input_output_assign.cpp
    main.cpp
    op.cpp
    graph_partition.cpp
    nop_elimination.cpp
    pass_liveness.cpp
    pass_manager.cpp
    pass_memory_layout.cpp
    serialize.cpp
    pattern.cpp
    shape.cpp
    reshape_elimination.cpp
    tensor.cpp
    type_prop.cpp
    util.cpp
    uuid.cpp
    zero_dim_tensor_elimination.cpp
)

if (NGRAPH_INTERPRETER_ENABLE)
    set(SRC ${SRC} backend_debug_api.cpp builder.cpp backend_api.cpp)
endif()

add_subdirectory(models)
add_subdirectory(files)
add_subdirectory(util)

if(NGRAPH_CPU_ENABLE)
    set(SRC ${SRC} backend_performance.cpp codegen.cpp cpu_fusion.cpp cpu_test.cpp)
endif()

if(NGRAPH_GPU_ENABLE)
    set(SRC ${SRC} cudnn.cpp gpu_test.cpp)
endif()

foreach(TEST_CONFIG ${UNIT_TEST_CONFIG_LIST})
    string(FIND ${TEST_CONFIG} "@" OFFSET)
    string(SUBSTRING ${TEST_CONFIG} 0 ${OFFSET} BACKEND_NAME)
    math(EXPR OFFSET ${OFFSET}+1)
    string(SUBSTRING ${TEST_CONFIG} ${OFFSET} -1 MANIFEST)

    configure_file(backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
    configure_file(convolution_test.in.cpp convolution_test_${BACKEND_NAME}.cpp)
    set(SRC ${CMAKE_CURRENT_BINARY_DIR}/backend_test_${BACKEND_NAME}.cpp ${SRC})
    set(SRC ${CMAKE_CURRENT_BINARY_DIR}/convolution_test_${BACKEND_NAME}.cpp ${SRC})
    if(NGRAPH_DISTRIBUTED_ENABLE)
        configure_file(distributed.cpp distributed_${BACKEND_NAME}.cpp)
        set(SRC ${CMAKE_CURRENT_BINARY_DIR}/distributed_${BACKEND_NAME}.cpp  ${SRC})
    endif()

    # Some---but not all---autodiff tests go through multiple iterations with
    # different random seeds. On the CPU backend this is currently very slow
    # because the autodiff tests recompile with each iteration. That behavior
    # can be changed, but it's a bit involved, so for the time being we just
    # reduce the number of test iterations on non-INTERPRETER backends.
    if(${BACKEND_NAME} MATCHES ^INTERPRETER$)
        set(TEST_LOOPS 100)
    else()
        set(TEST_LOOPS 2)
    endif()

    configure_file(autodiff.in.cpp autodiff_${BACKEND_NAME}.cpp)
    set(SRC ${CMAKE_CURRENT_BINARY_DIR}/autodiff_${BACKEND_NAME}.cpp ${SRC})

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

include_directories(".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJSON_INCLUDES=\\\"${JSON_INCLUDE_DIR}\\\"")

if(NGRAPH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    add_definitions(-DNGRAPH_DISTRIBUTED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    link_libraries(${MPI_CXX_LIBRARIES})
endif()

add_executable(unit-test ${SRC})
target_link_libraries(unit-test ngraph_test_util)
target_link_libraries(unit-test ngraph libgtest libjson pthread)
target_link_libraries(unit-test ${CMAKE_DL_LIBS})

if (NGRAPH_CPU_ENABLE)
    # The INTERPRETER backend is required for graph_partition, convolution, and backwards unit tests
    target_link_libraries(unit-test cpu_backend interpreter_backend)
    target_link_libraries(unit-test libmkldnn)
endif()

if (NGRAPH_TBB_ENABLE)
    add_definitions(-DNGRAPH_TBB_ENABLE)
endif()

if (NGRAPH_INTERPRETER_ENABLE)
    add_definitions(-DNGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(unit-test interpreter_backend)
endif()

if (NGRAPH_GPU_ENABLE)
    target_link_libraries(unit-test gpu_backend)
endif()

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
