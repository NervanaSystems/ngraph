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


from opgen.exception import ClassReadError


class OpArgument():
    def __init__(self, fields):
        self._init_from_fields(fields)

    def _init_from_fields(self, fields):
        if 'name' in fields:
            self.name = str(fields['name'])
        else:
            raise ClassReadError('Required field \'name\' is missing')

        if 'description' in fields:
            self.description = str(fields['description'])
        else:
            self.description = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    def __repr__(self):
        return('opgen.spec.OpArgument(name=%s, description=%s)' % (repr(self.name), repr(self.description)))


class OpResult():
    def __init__(self, fields):
        self._init_from_fields(fields)

    def _init_from_fields(self, fields):
        if 'name' in fields:
            self.name = str(fields['name'])
        else:
            raise ClassReadError('Required field \'name\' is missing')

        if 'description' in fields:
            self.description = str(fields['description'])
        else:
            self.description = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    def __repr__(self):
        return('opgen.spec.OpResult(name=%s, description=%s)' % (repr(self.name), repr(self.description)))


class OpAttribute():
    def __init__(self, fields):
        self._init_from_fields(fields)

    def _init_from_fields(self, fields):
        if 'name' in fields:
            self.name = str(fields['name'])
        else:
            raise ClassReadError('Required field \'name\' is missing')

        if 'type' in fields:
            self.type = str(fields['type'])
        else:
            raise ClassReadError('Required field \'type\' is missing')

        if 'description' in fields:
            self.description = str(fields['description'])
        else:
            self.description = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        self._type = val

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    def __repr__(self):
        return('opgen.spec.OpAttribute(name=%s, type=%s, description=%s)' % (repr(self.name), repr(self.type), repr(self.description)))


class OpClass():
    def __init__(self, fields):
        self._init_from_fields(fields)

    def _init_from_fields(self, fields):
        if 'name' in fields:
            self.name = str(fields['name'])
        else:
            raise ClassReadError('Required field \'name\' is missing')

        if 'dialect' in fields:
            self.dialect = str(fields['dialect'])
        else:
            raise ClassReadError('Required field \'dialect\' is missing')

        if 'description' in fields:
            self.description = str(fields['description'])
        else:
            self.description = ''

        self.commutative = 'commutative' in fields and fields['commutative']
        self.has_state = 'has_state' in fields and fields['has_state']
        self.validation_implemented = 'validation_implemented' in fields and fields[
            'validation_implemented']
        self.adjoints_implemented = 'adjoints_implemented' in fields and fields[
            'adjoints_implemented']
        self.zdte_implemented = 'zdte_implemented' in fields and fields[
            'zdte_implemented']

        self.arguments = []
        if 'arguments' in fields:
            for argument in fields['arguments']:
                self.arguments.append(OpArgument(argument))

        self.results = []
        if 'results' in fields:
            for result in fields['results']:
                self.results.append(OpResult(result))

        self.attributes = []
        if 'attributes' in fields:
            for attribute in fields['attributes']:
                self.attributes.append(OpAttribute(attribute))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def dialect(self):
        return self._dialect

    @dialect.setter
    def dialect(self, val):
        self._dialect = val

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    @property
    def commutative(self):
        return self._commutative

    @commutative.setter
    def commutative(self, val):
        self._commutative = val

    @property
    def has_state(self):
        return self._has_state

    @has_state.setter
    def has_state(self, val):
        self._has_state = val

    @property
    def validation_implemented(self):
        return self._validation_implemented

    @validation_implemented.setter
    def validation_implemented(self, val):
        self._validation_implemented = val

    @property
    def adjoints_implemented(self):
        return self._adjoints_implemented

    @adjoints_implemented.setter
    def adjoints_implemented(self, val):
        self._adjoints_implemented = val

    @property
    def zdte_implemented(self):
        return self._zdte_implemented

    @zdte_implemented.setter
    def zdte_implemented(self, val):
        self._zdte_implemented = val

    @property
    def arguments(self):
        return self._arguments

    @arguments.setter
    def arguments(self, val):
        self._arguments = val

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, val):
        self._results = val

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        self._attributes = val

    def __repr__(self):
        return('opgen.spec.OpClass(name=%s, dialect=%s, description=%s, arguments=%s, results=%s, attributes=%s)' % (repr(self.name), repr(self.dialect), repr(self.description), repr(self.arguments), repr(self.results), repr(self.attributes)))
