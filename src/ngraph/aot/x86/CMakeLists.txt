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

if (NOT NGRAPH_X86_AOT_ENABLE)
    return()
endif()

set(SRC
    ngraph_version.cpp
)

add_library(x86_aot shared ${SRC})
set_target_properties( x86_aot
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${NGRAPH_BUILD_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NGRAPH_BUILD_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${NGRAPH_BUILD_DIR}"
)

#target_compile_definitions(cpu_backend PRIVATE "NGRAPH_DEX_ONLY")
#ExternalProject_Get_Property(ext_halide BINARY_DIR)
#target_include_directories(cpu_backend SYSTEM PRIVATE ${BINARY_DIR}/include)
#target_link_libraries(cpu_backend PRIVATE ${BINARY_DIR}/lib/libHalide.a libhalidellvm)
#add_dependencies(cpu_backend ext_halide)

add_dependencies(codegen)
install(TARGETS x86_aot LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
