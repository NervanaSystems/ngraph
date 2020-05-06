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

from ngraph.impl import Dimension


def test_dimension():
    dim = Dimension()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == '<Dimension: ?>'

    dim = Dimension.dynamic()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == '<Dimension: ?>'

    dim = Dimension(10)
    assert dim.is_static
    assert len(dim) == 10
    assert dim.get_length() == 10
    assert dim.get_min_length() == 10
    assert dim.get_max_length() == 10
    assert repr(dim) == '<Dimension: 10>'

    dim = Dimension(5, 15)
    assert dim.is_dynamic
    assert dim.get_min_length() == 5
    assert dim.get_max_length() == 15
    assert repr(dim) == '<Dimension: [5, 15]>'


def test_dimension_comparisons():
    d1 = Dimension.dynamic()
    d2 = Dimension.dynamic()
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension.dynamic()
    d2 = Dimension(3)
    assert not d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert not d2.relaxes(d1)
    assert d2.compatible(d1)
    assert not d2.same_scheme(d1)

    d1 = Dimension(3)
    d2 = Dimension(3)
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension(4)
    d2 = Dimension(3)
    assert not d1.refines(d2)
    assert not d1.relaxes(d2)
    assert not d2.refines(d1)
    assert not d2.relaxes(d1)
    assert not d2.compatible(d1)
    assert not d2.same_scheme(d1)

