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
import numpy as np
import pytest

import ngraph as ng
from test.ngraph.util import get_runtime


@pytest.mark.skip_on_gpu
def test_variadic_split():
    runtime = get_runtime()
    input_tensor = ng.constant(np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.int32))
    axis = ng.constant(1, dtype=np.int64)
    splits = ng.constant(np.array([2, 4], dtype=np.int64))

    op = ng.variadic_split(input_tensor, axis, splits)
    model0 = runtime.computation(ng.get_output_element(op, 0))
    result0 = model0()
    split0 = np.array([[0, 1], [6, 7]], dtype=np.int32)
    assert np.allclose(result0, split0)

    model1 = runtime.computation(ng.get_output_element(op, 1))
    result1 = model1()
    split1 = np.array([[2, 3, 4, 5], [8, 9, 10, 11]], dtype=np.int32)
    assert np.allclose(result1, split1)
