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

if(MKLDNN_INCLUDE_DIR)
    link_directories(${MKLDNN_LIB_DIR})
endif()

if (NGRAPH_CPU_ENABLE)

    set(HEADER_SEARCH_DEFINES
        "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    )

    # abc
    set (ABC_SRC
        ${PROJECT_SOURCE_DIR}/doc/examples/abc.cpp
    )

    set_source_files_properties(${ABC_SRC} PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
    add_executable(abc ${ABC_SRC})
    add_dependencies(abc ngraph)
    target_link_libraries(abc ngraph)

    # abc_op
    set (ABC_OP_SRC
        ${PROJECT_SOURCE_DIR}/doc/examples/abc_operator.cpp
    )

    set_source_files_properties(${ABC_OP_SRC} PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
    add_executable(abc_op ${ABC_OP_SRC})
    add_dependencies(abc_op ngraph)
    target_link_libraries(abc_op ngraph)

    # update
    set (UPDATE_SRC
        ${PROJECT_SOURCE_DIR}/doc/examples/update.cpp
    )

    set_source_files_properties(${UPDATE_SRC} PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
    add_executable(update ${UPDATE_SRC})
    add_dependencies(update ngraph)
    target_link_libraries(update ngraph)

    # mnist_mlp
    set(MNIST_SRC
        ${PROJECT_SOURCE_DIR}/doc/examples/mnist_loader.cpp 
        ${PROJECT_SOURCE_DIR}/doc/examples/mnist_mlp.cpp
    )

    set_source_files_properties(${MNIST_SRC} PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
    add_executable(mnist_mlp ${MNIST_SRC})
    add_dependencies(mnist_mlp ngraph)
    target_link_libraries(mnist_mlp ngraph)

endif()
