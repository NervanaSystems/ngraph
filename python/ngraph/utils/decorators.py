# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
from functools import wraps
from typing import Any, Callable

from ngraph.impl import Node
from ngraph.utils.types import as_node, NodeInput
from ngraph.utils.broadcasting import as_elementwise_compatible_nodes


def _set_node_name(node, **kwargs):  # type: (Node, **Any) -> Node
    if 'name' in kwargs:
        node.name = kwargs['name']
    return node


def nameable_op(node_factory_function):  # type: (Callable) -> Callable
    """Set the name to the ngraph operator returned by the wrapped function."""
    @wraps(node_factory_function)
    def wrapper(*args, **kwargs):  # type: (*Any, **Any) -> Node
        node = node_factory_function(*args, **kwargs)
        node = _set_node_name(node, **kwargs)
        return node
    return wrapper


def unary_op(node_factory_function):  # type: (Callable) -> Callable
    """Convert the first input value to a Constant Node if a scalar value is detected."""
    @wraps(node_factory_function)
    def wrapper(input_value, *args, **kwargs):  # type: (NodeInput, *Any, **Any) -> Node
        input_node = as_node(input_value)
        node = node_factory_function(input_node, *args, **kwargs)
        node = _set_node_name(node, **kwargs)
        return node
    return wrapper


def binary_op(node_factory_function):  # type: (Callable) -> Callable
    """Convert the first two input values to Constant Nodes if scalar values are detected."""
    @wraps(node_factory_function)
    def wrapper(left, right, *args, **kwargs):  # type: (NodeInput, NodeInput, *Any, **Any) -> Node
        left, right = as_elementwise_compatible_nodes(left, right)
        node = node_factory_function(left, right, *args, **kwargs)
        node = _set_node_name(node, **kwargs)
        return node
    return wrapper
