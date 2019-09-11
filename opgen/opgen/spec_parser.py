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
"""Class specifications"""


import opgen.spec as spec
from opgen.spec import (OpArgument, OpResult, OpAttribute, OpClass)

import json

class SpecParser():
    def __init__(self):
        pass

    def parse_argument(self,j):
        if isinstance(j,str):
            j = json.loads(j)

        return OpArgument(**j)

    def parse_result(self,j):
        if isinstance(j,str):
            j = json.loads(j)

        return OpResult(**j)

    def parse_attribute(self,j):
        if isinstance(j,str):
            j = json.loads(j)

        return OpAttribute(**j)

    def parse_class(self,j):
        if isinstance(j,str):
            j = json.loads(j)

        if 'arguments' in j:
            args_parsed = []
            for arg_j in j['arguments']:
                args_parsed.append(self.parse_argument(arg_j))
            j['arguments'] = args_parsed

        if 'results' in j:
            results_parsed = []
            for res_j in j['results']:
                results_parsed.append(self.parse_result(res_j))
            j['results'] = results_parsed

        if 'attributes' in j:
            attributes_parsed = []
            for attr_j in j['attributes']:
                attributes_parsed.append(self.parse_attribute(attr_j))
            j['attributes'] = attributes_parsed

        return OpClass(**j)
