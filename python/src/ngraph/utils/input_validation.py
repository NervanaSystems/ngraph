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

"""Helper functions for validating user input."""

import logging
import numpy as np
from typing import Any, Callable, Iterable, Optional, Type

from ngraph.exceptions import UserInputError

log = logging.getLogger(__name__)


def assert_list_of_ints(value_list, message):  # type: (Iterable[int], str) -> None
    """Verify that the provided value is an iterable of integers."""
    try:
        for value in value_list:
            if not isinstance(value, int):
                raise TypeError
    except TypeError:
        log.warning(message)
        raise UserInputError(message, value_list)


def _check_value(attr_key, value, val_type, cond=None):
    # type: (str, Any, Type, Optional[Callable[[Any], bool]]) -> bool
    """Check whether provided value satisfies specified criteria.

    :param      attr_key:        The attribute name.
    :param      value:           The value to check.
    :param      val_type:        Required value type.
    :param      cond:            The optional function running additional checks.

    :raises     UserInputError:
    :return:    True if attribute satisfies all criterias. Otherwise False.
    """
    if not np.issubdtype(type(value), val_type):
        raise UserInputError(
            'Attribute \"{}\" value must by of type {}.'.format(attr_key, val_type))
    if cond is not None and not cond(value):
        raise UserInputError(
            'Attribute \"{}\" value does not satisfy provided condition.'.format(attr_key))
    return True


def check_valid_attribute(attr_dict, attr_key, val_type, cond=None, required=False):
    # type: (dict, str, Type, Optional[Callable[[Any], bool]], Optional[bool]) -> bool
    """Check whether specified attribute satisfies given criteria.

    :param attr_dict:   Dictionary containing key-value attributes to check.
    :param attr_key:    Key value for validated attribute.
    :param val_type:    Value type for validated attribute.
    :param cond:        Any callable wich accept attribute value and returns True or False.
    :param required:    Whether provided attribute key is not required. This mean it may be missing
                        from provided dictionary.

    :raises     UserInputError:

    :return: True if attribute satisfies all criterias. Otherwise False.
    """
    result = True

    if required and attr_key not in attr_dict:
        raise UserInputError('Provided dictionary is missing required attribute \"{}\"'.format(
            attr_key))

    if attr_key not in attr_dict:
        return result

    attr_value = attr_dict[attr_key]

    if np.isscalar(attr_value):
        result = result and _check_value(attr_key, attr_value, val_type, cond)
    else:
        for v in attr_value:
            result = result and _check_value(attr_key, v, val_type, cond)

    return result
