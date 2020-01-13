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

find_package(Doxygen REQUIRED)

if ("${NGRAPH_DOXYGEN_WARN_IF_UNDOCUMENTED}" MATCHES "^ON$")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
else()
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
endif()

if ("${NGRAPH_DOXYGEN_QUIET}" MATCHES "^ON$")
    set(DOXYGEN_QUIET YES)
else()
    set(DOXYGEN_QUIET NO)
endif()

set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
configure_file("${DOXYGEN_IN}" "${DOXYGEN_OUT}" @ONLY)

add_custom_target(doxygen-docs
    ALL
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating documentation with Doxygen"
    VERBATIM )

add_dependencies( docs doxygen-docs )

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
    DESTINATION "${NGRAPH_INSTALL_DOC}/api-reference/html"
    OPTIONAL
    )

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/latex/"
    DESTINATION "${NGRAPH_INSTALL_DOC}/api-reference/latex"
    OPTIONAL
    )
