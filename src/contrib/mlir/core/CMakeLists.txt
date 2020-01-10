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

include_directories(
    ${NGRAPH_INCLUDE_PATH}
    ${MLIR_LLVM_INCLUDE_PATH}
    ${MLIR_INCLUDE_PATHS}
)

set(SRC
    compiler.cpp
    ngraph_dialect/dialect.cpp
    ngraph_dialect/type.cpp
    ngraph_dialect/ops.cpp
    pass/mlir_subgraph_extraction.cpp
    pass/mlir_subgraph_extraction.hpp
    pass/ng_dialect_builder.cpp
    pass/ng_dialect_builder.hpp
)

add_library(mlir_core SHARED ${SRC})
install(TARGETS mlir_core DESTINATION ${NGRAPH_INSTALL_LIB})
