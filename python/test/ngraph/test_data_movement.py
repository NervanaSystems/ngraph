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

import ngraph as ng
from test.ngraph.util import get_runtime


def test_reverse_sequence():
    input_data = np.array(
        [
            0, 0, 3, 0, 6, 0, 9, 0, 1, 0, 4, 0, 7, 0, 10, 0, 2,
            0, 5, 0, 8, 0, 11, 0, 12, 0, 15, 0, 18, 0, 21, 0,
            13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
        ],
        dtype=np.int32,
    ).reshape([2, 3, 4, 2])
    seq_lenghts = np.array([1, 2, 1, 2], dtype=np.int32)
    batch_axis = 2
    sequence_axis = 1

    input_param = ng.parameter(input_data.shape, name='input', dtype=np.int32)
    seq_lengths_param = ng.parameter(seq_lenghts.shape, name='sequence lengths', dtype=np.int32)
    model = ng.reverse_sequence(input_param, seq_lengths_param, batch_axis, sequence_axis)

    runtime = get_runtime()
    computation = runtime.computation(model, input_param, seq_lengths_param)
    result = computation(input_data, seq_lenghts)

    expected = np.array(
        [
            0, 0, 4, 0, 6, 0, 10, 0, 1, 0, 3, 0, 7, 0, 9, 0, 2,
            0, 5, 0, 8, 0, 11, 0, 12, 0, 16, 0, 18, 0, 22, 0, 13,
            0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0,
        ],
    ).reshape([1, 2, 3, 4, 2])
    assert np.allclose(result, expected)
