# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Factory functions for all ngraph ops."""

import numpy as np

from pyngraph.op import Parameter

from ngraph_api.utils.input_validation import assert_list_of_ints
from ngraph_api.utils.types import get_element_type, py_numeric_type, tensor_shape
from ngraph_api.utils import nameable_op


@nameable_op
def parameter(shape, dtype=np.float32, name=None):
    # type: (tensor_shape, py_numeric_type, str) -> Parameter
    """Return an ngraph Parameter object."""
    assert_list_of_ints(shape, 'Parameter shape must be a list of integer values.')
    element_type = get_element_type(dtype)
    return Parameter(element_type, shape)
