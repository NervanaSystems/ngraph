# ******************************************************************************
# Copyright 2018 Intel Corporation
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
    plaidml_backend.cpp
    plaidml_builder.cpp
    plaidml_compilation_cache.cpp
    plaidml_compiled_function.cpp
    plaidml_compiler.cpp
    plaidml_config.cpp
    plaidml_convpool_formatter.cpp
    plaidml_impl.cpp
    plaidml_logger.cpp
    plaidml_ops_arithmetic.cpp
    plaidml_ops_batch_norm.cpp
    plaidml_ops_comparison.cpp
    plaidml_ops_concat.cpp
    plaidml_ops_convert.cpp
    plaidml_ops_convolution.cpp
    plaidml_ops_dot.cpp
    plaidml_ops_function.cpp
    plaidml_ops_general.cpp
    plaidml_ops_index_reduction.cpp
    plaidml_ops_io.cpp
    plaidml_ops_local_response_norm.cpp
    plaidml_ops_logical.cpp
    plaidml_ops_one_hot.cpp
    plaidml_ops_pool.cpp
    plaidml_ops_reduce.cpp
    plaidml_ops_replace_slice.cpp
    plaidml_ops_reverse.cpp
    plaidml_ops_slice.cpp
    plaidml_ops_softmax.cpp
    plaidml_ops_transcendental.cpp
    plaidml_tensor.cpp
    plaidml_translate.cpp
)

if (NGRAPH_PLAIDML_ENABLE)
    find_package(PlaidML CONFIG REQUIRED)
    message(STATUS "PlaidML enabled")

    add_library(libplaidml INTERFACE)
    target_link_libraries(libplaidml INTERFACE ${PLAIDML_LIBRARIES})
    install(FILES ${PLAIDML_LIBRARIES} DESTINATION ${NGRAPH_INSTALL_LIB})

    add_library(plaidml_backend SHARED ${SRC})
    set_target_properties(plaidml_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_include_directories(plaidml_backend SYSTEM PUBLIC ${PLAIDML_INCLUDE_DIRS})
    target_link_libraries(plaidml_backend PUBLIC ngraph libplaidml)
    set_target_properties(plaidml_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${NGRAPH_BUILD_DIR}")
    install(TARGETS plaidml_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
else()
    message(STATUS "PlaidML not enabled")
endif()
  
