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


class OpArgument():
    def __init__(self, name, description=''):
        self.name = name
        self.description = description

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = str(val)

    def __repr__(self):
        return('opgen.spec.OpArgument(name=%s, description=%s)' % (repr(self.name), repr(self.description)))


class OpResult():
    def __init__(self, name, description=''):
        self.name = name
        self.description = description

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = str(val)

    def __repr__(self):
        return('opgen.spec.OpResult(name=%s, description=%s)' % (repr(self.name), repr(self.description)))


class OpAttribute():
    def __init__(self, name, type, description=''):
        self.name = name
        self.type = type
        self.description = description

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        self._type = str(val)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = str(val)

    def __repr__(self):
        return('opgen.spec.OpAttribute(name=%s, type=%s, description=%s)' % (repr(self.name), repr(self.type), repr(self.description)))


class OpClass():
    def __init__(self, name, dialect, description='', arguments=[], results=[], attributes=[], commutative=False, has_state=False, validation_implemented=False, adjoints_implemented=False, zdte_implemented=False):
        self.name = name
        self.dialect = dialect
        self.description = description
        self.arguments = arguments
        self.results = results
        self.attributes = attributes
        self.commutative = commutative
        self.has_state = has_state
        self.validation_implemented = validation_implemented
        self.adjoints_implemented = adjoints_implemented
        self.zdte_implemented = zdte_implemented

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def dialect(self):
        return self._dialect

    @dialect.setter
    def dialect(self, val):
        self._dialect = str(val)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = str(val)

    @property
    def commutative(self):
        return self._commutative

    @commutative.setter
    def commutative(self, val):
        self._commutative = bool(val)

    @property
    def has_state(self):
        return self._has_state

    @has_state.setter
    def has_state(self, val):
        self._has_state = bool(val)

    @property
    def validation_implemented(self):
        return self._validation_implemented

    @validation_implemented.setter
    def validation_implemented(self, val):
        self._validation_implemented = bool(val)

    @property
    def adjoints_implemented(self):
        return self._adjoints_implemented

    @adjoints_implemented.setter
    def adjoints_implemented(self, val):
        self._adjoints_implemented = bool(val)

    @property
    def zdte_implemented(self):
        return self._zdte_implemented

    @zdte_implemented.setter
    def zdte_implemented(self, val):
        self._zdte_implemented = bool(val)

    @property
    def arguments(self):
        return self._arguments

    @arguments.setter
    def arguments(self, val):
        assert(isinstance(val, list))
        assert(all(isinstance(arg, OpArgument) for arg in val))
        self._arguments = val

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, val):
        assert(isinstance(val, list))
        assert(all(isinstance(res, OpResult) for res in val))
        self._results = val

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        assert(isinstance(val, list))
        assert(all(isinstance(attr, OpAttribute) for attr in val))
        self._attributes = val

    def __repr__(self):
        return('opgen.spec.OpClass(name=%s, dialect=%s, description=%s,'
               'arguments=%s, results=%s, attributes=%s,'
               'commutative=%s, has_state=%s, adjoints_implemented=%s,'
               'validation_implemented=%s, zdte_implemented=%s)'
               % (repr(self.name), repr(self.dialect), repr(self.description),
                  repr(self.arguments), repr(
                      self.results), repr(self.attributes),
                  repr(self.commutative), repr(self.has_state), repr(
                      self.adjoints_implemented),
                  repr(self.validation_implemented), repr(self.zdte_implemented)))
