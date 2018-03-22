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
from __future__ import print_function, division

import numpy as np
import pytest

import ngraph as ng


def test_avg_pooling_3d():
    manager_name = pytest.config.getoption('backend', default='CPU')
    rt = ng.runtime(manager_name=manager_name)

    data = np.arange(11, 27, dtype=np.float32)
    data = data.reshape((1, 1, 4, 4))
    data = np.broadcast_to(data, (1, 1, 4, 4, 4))

    param = ng.parameter(data.shape)

    avgpool = ng.avg_pool(param,
                          [2, 2, 2],
                          [2, 2, 2])
    comp = rt.computation(avgpool, param)
    result = comp(data)
    result_ref = [[[[[13.5, 15.5],
                     [21.5, 23.5]],

                    [[13.5, 15.5],
                     [21.5, 23.5]]]]]
    np.testing.assert_allclose(result, result_ref, rtol=0.001)
