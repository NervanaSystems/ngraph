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

"""Helper functions for validating user input."""

import logging
from typing import Iterable

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
