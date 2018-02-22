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

include_directories(
    SYSTEM
    ${GTEST_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    )

include_directories(
    ${NGRAPH_INCLUDE_DIR}
    )

set (SRC
    backend_debug_api.cpp
    builder.cpp
    builder_autobroadcast.cpp
    builder_xla.cpp
    build_graph.cpp
    copy.cpp
    core_fusion.cpp
    eigen.cpp
    element_type.cpp
    file_util.cpp
    inliner.cpp
    input_output_assign.cpp
    main.cpp
    op.cpp
    pass_liveness.cpp
    pass_manager.cpp
    pass_memory_layout.cpp
    runtime_manager.cpp
    serialize.cpp
    pattern.cpp
    shape.cpp
    reshape_elimination.cpp
    tensor.cpp
    type_prop.cpp
    util/autodiff/backprop_function.cpp
    util/test_tools.cpp
    util/benchmark.cpp
    util.cpp
    uuid.cpp
)

add_subdirectory(models)

#================================================================================================
# To auto generate a suite of unit tests for a backend add a line like this
# set(BACKEND_NAMES ${BACKEND_NAMES} "BACKEND_NAME_GOES_HERE")
# and replace BACKEND_NAME_GOES_HERE with your backend name.
# The code for the unit test suite is in test/backend_test.in.cpp
#================================================================================================
# TODO add interpreter back to unit tests when it works
set(BACKEND_NAMES ${BACKEND_NAMES} "INTERPRETER")

if(MKLDNN_INCLUDE_DIR)
    include_directories(SYSTEM ${MKLDNN_INCLUDE_DIR})
    link_directories(${MKLDNN_LIB_DIR})
    set(SRC ${SRC} mkldnn.cpp)
endif()

if(NGRAPH_CPU_ENABLE AND LLVM_INCLUDE_DIR)
    include_directories(SYSTEM ${LLVM_INCLUDE_DIR})
    link_directories(${LLVM_LIB_DIR})
    set(SRC ${SRC} backend_performance.cpp codegen.cpp cpu_fusion.cpp)
    set(BACKEND_NAMES ${BACKEND_NAMES} "CPU")
endif()

if(NGRAPH_GPU_ENABLE AND LLVM_INCLUDE_DIR)
    include_directories(SYSTEM ${LLVM_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})    
    link_directories(${LLVM_LIB_DIR})
    link_directories(${CUDA_LIBRARIES})
    link_directories(${CUDA_CUBLAS_LIBRARIES})
    link_directories(${CUDNN_LIBRARIES})
    set(SRC
        ${SRC}
        cudnn.cpp)
    # Disabled for testing
    set(BACKEND_NAMES ${BACKEND_NAMES} "GPU")
endif()

if(NGRAPH_ARGON_ENABLE)
    include_directories(SYSTEM ${ARGON_TRANSFORMER_INCLUDE_DIR})
    set(BACKEND_NAMES ${BACKEND_NAMES} "ARGON")
    set(SRC ${SRC} argon_fusion.cpp)

    # enable additional Argon backend test 
    set(ARGON_ADDITIONAL_BACKEND_TESTS ${ARGON_TRANSFORMER_SOURCE_DIR}/test/test_argon_backend)
    message(STATUS "ARGON_ADDITIONAL_BACKEND_TESTS path: ${ARGON_ADDITIONAL_BACKEND_TESTS}")
    set(ADDITIONAL_ARGON_TEST
            ${ARGON_ADDITIONAL_BACKEND_TESTS}/test_reduce_sum.cpp
            ${ARGON_ADDITIONAL_BACKEND_TESTS}/test_dot.cpp
            ${ARGON_ADDITIONAL_BACKEND_TESTS}/test_broadcast.cpp
            ${ARGON_ADDITIONAL_BACKEND_TESTS}/test_conv.cpp
            ${ARGON_ADDITIONAL_BACKEND_TESTS}/test_binary_ew.cpp)
    
    # ensures ADDITIONAL_ARGON_TEST are a dependency on argon transformer
    add_custom_command(OUTPUT ${ADDITIONAL_ARGON_TEST} DEPENDS ext_argon_transformer COMMAND "")
    set(SRC ${SRC} ${ADDITIONAL_ARGON_TEST})
endif()

if(NGRAPH_DISTRIBUTED_ENABLE AND MPI_C_INCLUDE_PATH)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    foreach(BACKEND_NAME ${BACKEND_NAMES})
        configure_file(distributed.cpp distributed_${BACKEND_NAME}.cpp)
        set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/distributed_${BACKEND_NAME}.cpp)
endforeach()
endif()

foreach(BACKEND_NAME ${BACKEND_NAMES})
    configure_file(backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
    configure_file(convolution_test.in.cpp convolution_test_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/backend_test_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/convolution_test_${BACKEND_NAME}.cpp)

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
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/autodiff_${BACKEND_NAME}.cpp)

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

include_directories(".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")
if(NGRAPH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

add_executable(unit-test ${SRC})

if(MPI_C_INCLUDE_PATH)
    target_link_libraries(unit-test ${MPI_CXX_LIBRARIES})
endif()

if(MKLDNN_INCLUDE_DIR)
    target_link_libraries(unit-test mkldnn)
    add_dependencies(unit-test ext_mkldnn)
endif()

if(LLVM_INCLUDE_DIR)
    target_link_libraries(unit-test ${LLVM_LINK_LIBS})
    add_dependencies(unit-test ext_llvm)
endif()

if(CUDA_INCLUDE_DIRS)
    target_link_libraries(unit-test ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

target_link_libraries(unit-test ngraph libgtest pthread)
target_link_libraries(unit-test ${CMAKE_DL_LIBS})

add_dependencies(unit-test ngraph libgtest ext_eigen)

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

