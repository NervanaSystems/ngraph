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
from test.ngraph.util import run_op_node


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
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=reduction_axes)
    result = run_op_node([input_data], ng_api_helper, reduction_axes)
    assert np.allclose(result, expected)
