# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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
from test.ngraph.util import get_runtime


@pytest.config.gpu_skip(reason='Not implemented')
def test_lrn():
    input_image_shape = (2, 3, 2, 1)
    input_image = np.arange(int(np.prod(input_image_shape))).reshape(input_image_shape).astype('f')

    runtime = get_runtime()
    model = ng.lrn(ng.constant(input_image), alpha=1.0, beta=2.0, bias=1.0, size=3)
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(result,
                       np.array([[[[0.0],
                                   [0.05325444]],
                                  [[0.03402646],
                                   [0.01869806]],
                                  [[0.06805293],
                                   [0.03287071]]],
                                 [[[0.00509002],
                                   [0.00356153]],
                                  [[0.00174719],
                                   [0.0012555]],
                                  [[0.00322708],
                                   [0.00235574]]]], dtype=np.float32))

    # Test LRN default parameter values
    model = ng.lrn(ng.constant(input_image))
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(result,
                       np.array([[[[0.0],
                                   [0.35355338]],
                                  [[0.8944272],
                                   [1.0606602]],
                                  [[1.7888544],
                                   [1.767767]]],
                                 [[[0.93704253],
                                   [0.97827977]],
                                  [[1.2493901],
                                   [1.2577883]],
                                  [[1.5617375],
                                   [1.5372968]]]], dtype=np.float32))
