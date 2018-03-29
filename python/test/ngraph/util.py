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

import numpy as np
import pytest

import ngraph as ng


def get_runtime():
    """Return runtime object."""
    manager_name = pytest.config.getoption('backend', default='CPU')
    return ng.runtime(manager_name=manager_name)


def run_op_node(input_data, op_fun, *args):
    """Run computation on node performing `op_fun`.

    `op_fun` have to needs to accept a node as an argument.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    parameter_a = ng.parameter(input_data.shape, name='A', dtype=np.float32)
    node = op_fun(parameter_a, *args)
    computation = runtime.computation(node, parameter_a)
    return computation(input_data)


def run_op_numeric_data(input_data, op_fun, *args):
    """Run computation on node performing `op_fun`.

    `op_fun` have to accept a scalar or an array.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    node = op_fun(input_data, *args)
    computation = runtime.computation(node)
    return computation()
