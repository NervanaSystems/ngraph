#*******************************************************************************
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
#*******************************************************************************
"""Generic utilities. Factor related functions out to separate files."""

from functools import wraps
from typing import Callable, Any, Optional, Set
from ngraph_api.utils.types import TensorShape


def nameable_op(op_factory_function):  # type: (Callable) -> Callable
    """Set the name to the ngraph operator returned by the wrapped function."""
    @wraps(op_factory_function)
    def wrapper(*args, **kwds):  # type: (Any, Any) -> Any
        op = op_factory_function(*args, **kwds)
        if 'name' in kwds:
            op.name = kwds['name']
        return op
    return wrapper


def get_broadcast_axes(left_shape, right_shape, axis):
    # type: (TensorShape, TensorShape, Optional[int]) -> Set[int]
    """Generate a list of broadcast axes for ngraph++ broadcast.

    Informally, a broadcast "adds" axes to the input tensor,
    replicating elements from the input tensor as needed to fill the new dimensions.
    Function calculate which of the output axes is being so added.
    For example, an output shape of `{2,5,6,2,8}` and input shape of `{2,6}` means
    that the broadcast axes must be `{1,3,4}`.
    """
    axes_indexes = list(range(0, len(left_shape)))
    if(axis is None):
        right_begin = len(left_shape) - len(right_shape)
    else:
        right_begin = axis
    right_axes_indexes = list(range(right_begin, right_begin + len(right_shape)))
    for index in reversed(right_axes_indexes):
        del axes_indexes[index]
    return set(axes_indexes)
