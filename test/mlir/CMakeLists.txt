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

# Enable use of the lit tool that we build from MLIR repo.
set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
set(LLVM_DEFAULT_EXTERNAL_LIT ${MLIR_TOOLS_DIR}/llvm-lit)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )

set(NGRAPH_MLIR_TEST_DEPENDS
    ngraph-opt
    )

add_lit_testsuite(check-mlir-lit "Running the nGraph MLIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${NGRAPH_MLIR_TEST_DEPENDS}
    )
set_target_properties(check-mlir-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(NGRAPH_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${NGRAPH_MLIR_TEST_DEPS}
    )
