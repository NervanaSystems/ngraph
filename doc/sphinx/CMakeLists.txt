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

set(NGRAPH_BUILD_SPHINX_DOCS FALSE
    CACHE BOOL
    "The nGraph build system shall contain a target for '.rst'-based overview docs."
    )

if (NGRAPH_BUILD_SPHINX_DOCS)
    find_package(Sphinx REQUIRED)

    set(SPHINX_IN ${CMAKE_CURRENT_SOURCE_DIR}/source)
    set(SPHINX_OUT ${CMAKE_CURRENT_BINARY_DIR}/build)

    add_custom_target(sphinx-docs
        ALL
        COMMAND ${SPHINX_EXECUTABLE} -b html "${SPHINX_IN}" "${SPHINX_OUT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating documentation with Sphinx"
        VERBATIM )

    add_dependencies( docs sphinx-docs )
    set(DOCS_TARGET_IS_EMPTY FALSE PARENT_SCOPE)

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build/"
        DESTINATION "${NGRAPH_INSTALL_DOC}/overview/html"
        OPTIONAL
        )
endif()

