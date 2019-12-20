#*****************************************************************************
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
#*****************************************************************************
"""Lit runner configuration."""

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'nGraph MLIR Compiler'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# test_source_root: The root path where tests are located.
config.test_source_root = config.ngraph_mlir_test_src_dir

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.ngraph_mlir_test_build_dir

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.ngraph_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir
]
tool_names = [
    'ngraph-opt', 'mlir-opt', 'mlir-translate'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)
