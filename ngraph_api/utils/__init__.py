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
"""Generic utilities. Factor related functions out to separate files."""

from functools import wraps
from typing import Callable, Any


def nameable_op(op_factory_function):  # type: (Callable) -> Callable
    """Set the name to the ngraph operator returned by the wrapped function."""
    @wraps(op_factory_function)
    def wrapper(*args, **kwds):  # type: (Any, Any) -> Any
        op = op_factory_function(*args, **kwds)
        if 'name' in kwds:
            op.name = kwds['name']
        return op
    return wrapper
