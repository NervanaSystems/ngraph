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

get_filename_component( NGRAPH_INCLUDE_DIR . ABSOLUTE)
set(NGRAPH_INCLUDE_DIR "${NGRAPH_INCLUDE_DIR}" PARENT_SCOPE)

set (SRC
    element_type.cpp
    names.cpp
    strides.cpp
    tree.cpp
    util.cpp
    log.cpp

    transformers/axes.cpp
    transformers/exop.cpp
    transformers/mock_transformer.cpp
    transformers/ndarray.cpp
    transformers/op_graph.cpp

    values/function.cpp
    values/op.cpp
)

# NOTE: We'd prefer to only have the .cpp files *in* the 'transformers' directory be compiled
# with the 'transformers' directory in the include-path.  But it's hard to do that with
# CMake versions lower than 3.1: see
# https://stackoverflow.com/questions/9339851/can-one-add-further-source-files-to-an-executable-once-defined
include_directories(
    "${NGRAPH_INCLUDE_DIR}"
    "${NGRAPH_INCLUDE_DIR}/transformers"
    )

add_library(ngraph SHARED ${SRC})

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

set(DEPLOY_SRC_HEADERS
    element_type.hpp
    names.hpp
    strides.hpp
    tree.hpp
    util.hpp
    uuid.hpp
    )

set(DEPLOY_SRC_TRANSFORMERS_HEADERS
    transformers/axes.hpp
    transformers/exop.hpp
    transformers/mock.hpp
    transformers/mock_transformer.hpp
    transformers/ndarray.hpp
    transformers/op_graph.hpp
    )

install(TARGETS ngraph DESTINATION lib)
install(FILES ${DEPLOY_SRC_HEADERS} DESTINATION include/ngraph)
install(FILES ${DEPLOY_SRC_TRANSFORMERS_HEADERS} DESTINATION include/ngraph/transformers)

