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
import pytest
import test


def pytest_addoption(parser):
    parser.addoption('--backend', default='INTERPRETER',
                     choices=['INTERPRETER', 'CPU', 'GPU', 'NNP', 'PlaidML'],
                     help='Select from available backends')


def pytest_configure(config):
    backend_name = config.getvalue('backend')
    test.BACKEND_NAME = backend_name


def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue('backend')

    gpu_skip = pytest.mark.skip(reason='Skipping test on the GPU backend.')
    cpu_skip = pytest.mark.skip(reason='Skipping test on the CPU backend.')
    nnp_skip = pytest.mark.skip(reason='Skipping test on the NNP backend.')
    interpreter_skip = pytest.mark.skip(reason='Skipping test on the INTERPRETER backend.')
    plaidml_skip = pytest.mark.skip(reason='Skipping test on the PlaidML backend.')

    for item in items:
        if backend_name == 'GPU' and 'skip_on_gpu' in item.keywords:
            item.add_marker(gpu_skip)
        if backend_name == 'CPU' and 'skip_on_cpu' in item.keywords:
            item.add_marker(cpu_skip)
        if backend_name == 'NNP' and 'skip_on_nnp' in item.keywords:
            item.add_marker(nnp_skip)
        if backend_name == 'INTERPRETER' and 'skip_on_interpreter' in item.keywords:
            item.add_marker(interpreter_skip)
        if backend_name == 'PlaidML' and 'skip_on_plaidml' in item.keywords:
            item.add_marker(plaidml_skip)
