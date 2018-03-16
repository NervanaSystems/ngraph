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

import ngraph_api as ng


@pytest.mark.parametrize('ng_api_helper, numpy_function, reduction_axes', [
    (ng.max, np.max, None),
    (ng.min, np.min, None),
    (ng.sum, np.sum, None),
    (ng.prod, np.prod, None),
    (ng.max, np.max, (0, )),
    (ng.min, np.min, (0, )),
    (ng.sum, np.sum, (0, )),
    (ng.prod, np.prod, (0, )),
    (ng.max, np.max, (0, 2)),
    (ng.min, np.min, (0, 2)),
    (ng.sum, np.sum, (0, 2)),
    (ng.prod, np.prod, (0, 2)),
])
def test_reduction_ops(ng_api_helper, numpy_function, reduction_axes):
    manager_name = pytest.config.getoption('backend', default='CPU')
    runtime = ng.runtime(manager_name=manager_name)

    shape = [2, 4, 3, 2]
    parameter_a = ng.parameter(shape, name='A', dtype=np.float32)

    model = ng_api_helper(parameter_a, reduction_axes)
    computation = runtime.computation(model, parameter_a)

    value_a = np.random.randn(*shape).astype(np.float32)

    result = computation(value_a)
    expected = numpy_function(value_a, axis=reduction_axes)
    assert np.allclose(result, expected)
