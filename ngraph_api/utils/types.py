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
"""Functions related to converting between Python and numpy types and ngraph types"""
import logging
from typing import Union

import numpy as np

from ngraph_api.exceptions import NgraphTypeError
from pyngraph import Type as NgraphType


log = logging.getLogger(__file__)

py_numeric_type = Union[int, float, np.ndarray]

ngraph_to_numpy_types_map = [
    (NgraphType.f32, np.float32),
    (NgraphType.f64, np.float64),
    (NgraphType.i8,  np.int8),
    (NgraphType.i16, np.int16),
    (NgraphType.i32, np.int32),
    (NgraphType.i64, np.int64),
    (NgraphType.u8,  np.uint8),
    (NgraphType.u16, np.uint16),
    (NgraphType.u32, np.uint32),
    (NgraphType.u64, np.uint64)
]


def get_element_type(dtype: py_numeric_type) -> NgraphType:
    """Return an ngraph element type for a Python type or numpy.dtype."""
    if dtype == int:
        log.warning('Converting int type of undefined bitwidth to 32-bit ngraph integer.')
        return NgraphType.i32

    if dtype == float:
        log.warning('Converting float type of undefined bitwidth to 32-bit ngraph float.')
        return NgraphType.f32

    ng_type = next((ng_type for (ng_type, np_type) in ngraph_to_numpy_types_map if np_type == dtype), None)
    if ng_type:
        return ng_type

    raise NgraphTypeError('Unidentified data type %s', dtype)


def get_dtype(ngraph_type: NgraphType) -> type:
    np_type = next((np_type for (ng_type, np_type) in ngraph_to_numpy_types_map if ng_type == ngraph_type), None)

    if np_type:
        return np_type

    raise NgraphTypeError('Unidentified data type %s', np_type)
