# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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


def pytest_addoption(parser):
    parser.addoption('--backend', default='INTERPRETER',
                     choices=['INTERPRETER', 'CPU', 'GPU', 'NNP', 'PlaidML'],
                     help='Select from available backends')


def pass_method(*args, **kwargs):
    pass


def pytest_configure(config):
    config.gpu_skip = pytest.mark.skipif(config.getvalue('backend') == 'GPU')
    config.cpu_skip = pytest.mark.skipif(config.getvalue('backend') == 'CPU')
    config.nnp_skip = pytest.mark.skipif(config.getvalue('backend') == 'NNP')
    config.interpreter_skip = pytest.mark.skipif(config.getvalue('backend') == 'INTERPRETER')
    config.plaidml_skip = pytest.mark.skipif(config.getvalue('backend') == 'PlaidML')
