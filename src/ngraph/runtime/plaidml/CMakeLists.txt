# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
    plaidml_backend_constructor.cpp
    plaidml_builder.cpp
    plaidml_compilation_cache.cpp
    plaidml_compiler.cpp
    plaidml_config.cpp
    plaidml_convpool_formatter.cpp
    plaidml_executable.cpp
    plaidml_impl.cpp
    plaidml_logger.cpp
    plaidml_ops_arithmetic.cpp
    plaidml_ops_batch_norm.cpp
    plaidml_ops_comparison.cpp
    plaidml_ops_concat.cpp
    plaidml_ops_convert.cpp
    plaidml_ops_convolution.cpp
    plaidml_ops_dot.cpp
    plaidml_ops_general.cpp
    plaidml_ops_group_convolution.cpp
    plaidml_ops_implicit_broadcast.cpp
    plaidml_ops_index_reduction.cpp
    plaidml_ops_io.cpp
    plaidml_ops_local_response_norm.cpp
    plaidml_ops_logical.cpp
    plaidml_ops_one_hot.cpp
    plaidml_ops_passthrough.cpp
    plaidml_ops_pool.cpp
    plaidml_ops_quantize.cpp
    plaidml_ops_reduce.cpp
    plaidml_ops_replace_slice.cpp
    plaidml_ops_replicate.cpp
    plaidml_ops_reverse.cpp
    plaidml_ops_slice.cpp
    plaidml_ops_softmax.cpp
    plaidml_ops_transcendental.cpp
    plaidml_ops_winograd.cpp
    plaidml_pass_concat_elision.cpp
    plaidml_pass_concat_split.cpp
    plaidml_pass_explicit_logicals.cpp
    plaidml_pass_implicit_broadcast.cpp
    plaidml_pass_lower_convolutions.cpp
    plaidml_pass_replicate_combination.cpp
    plaidml_pass_replicate_elision.cpp
    plaidml_pass_winograd.cpp
    plaidml_tensor.cpp
    plaidml_translate.cpp
)

add_library(libplaidml INTERFACE)
target_link_libraries(libplaidml INTERFACE ${PLAIDML_LIBRARIES})
add_library(plaidml_backend SHARED ${SRC})
if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(plaidml_backend PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()
target_include_directories(plaidml_backend SYSTEM PUBLIC ${PLAIDML_INCLUDE_DIRS})
target_link_libraries(plaidml_backend PUBLIC ngraph libplaidml)
install(TARGETS plaidml_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})

target_compile_definitions(plaidml_backend PRIVATE PLAIDML_BACKEND_EXPORTS)

set(CMAKE_MACOSX_RPATH 1)
if(APPLE)
    set_property(TARGET plaidml_backend PROPERTY INSTALL_RPATH "@loader_path/;@loader_path/../../..")
elseif(DEFINED NGRAPH_RPATH)
    set_property(TARGET plaidml_backend PROPERTY INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..;${NGRAPH_RPATH}")
else()
    set_property(TARGET plaidml_backend PROPERTY INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..")
endif()
