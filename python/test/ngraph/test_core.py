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

from ngraph.impl import Dimension, PartialShape, Shape


def test_dimension():
    dim = Dimension()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == "<Dimension: ?>"

    dim = Dimension.dynamic()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == "<Dimension: ?>"

    dim = Dimension(10)
    assert dim.is_static
    assert len(dim) == 10
    assert dim.get_length() == 10
    assert dim.get_min_length() == 10
    assert dim.get_max_length() == 10
    assert repr(dim) == "<Dimension: 10>"

    dim = Dimension(5, 15)
    assert dim.is_dynamic
    assert dim.get_min_length() == 5
    assert dim.get_max_length() == 15
    assert repr(dim) == "<Dimension: [5, 15]>"


def test_dimension_comparisons():
    d1 = Dimension.dynamic()
    d2 = Dimension.dynamic()
    assert d1 == d2
    assert d1 == -1
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension.dynamic()
    d2 = Dimension(3)
    assert d1 != d2
    assert d2 == 3
    assert not d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert not d2.relaxes(d1)
    assert d2.compatible(d1)
    assert not d2.same_scheme(d1)

    d1 = Dimension(3)
    d2 = Dimension(3)
    assert d1 == d2
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension(4)
    d2 = Dimension(3)
    assert d1 != d2
    assert not d1.refines(d2)
    assert not d1.relaxes(d2)
    assert not d2.refines(d1)
    assert not d2.relaxes(d1)
    assert not d2.compatible(d1)
    assert not d2.same_scheme(d1)


def test_partial_shape():
    ps = PartialShape([1, 2, 3, 4])
    assert ps.is_static
    assert not ps.is_dynamic
    assert ps.rank == 4
    assert repr(ps) == "<PartialShape: {1,2,3,4}>"

    shape = Shape([1, 2, 3])
    ps = PartialShape(shape)
    assert ps.is_static
    assert not ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 3
    assert list(ps.get_shape()) == [1, 2, 3]
    assert list(ps.get_max_shape()) == [1, 2, 3]
    assert list(ps.get_min_shape()) == [1, 2, 3]
    assert list(ps.to_shape()) == [1, 2, 3]
    assert repr(shape) == "<Shape{1, 2, 3}>"
    assert repr(ps) == "<PartialShape: {1,2,3}>"

    ps = PartialShape([Dimension(1), Dimension(2), Dimension(3), Dimension.dynamic()])
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 4
    assert list(ps.get_min_shape()) == [1, 2, 3, 0]
    assert list(ps.get_max_shape())[3] > 1000000000
    assert repr(ps) == "<PartialShape: {1,2,3,?}>"

    ps = PartialShape([1, 2, 3, -1])
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 4
    assert list(ps.get_min_shape()) == [1, 2, 3, 0]
    assert list(ps.get_max_shape())[3] > 1000000000
    assert repr(ps) == "<PartialShape: {1,2,3,?}>"

    ps = PartialShape.dynamic()
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == Dimension.dynamic()
    assert list(ps.get_min_shape()) == []
    assert list(ps.get_max_shape()) == []
    assert repr(ps) == "<PartialShape: ?>"

    ps = PartialShape.dynamic(r=Dimension(2))
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == 2
    assert 2 == ps.rank
    assert list(ps.get_min_shape()) == [0, 0]
    assert list(ps.get_max_shape())[0] > 1000000000
    assert repr(ps) == "<PartialShape: {?,?}>"


def test_partial_shape_compatible():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.compatible(ps2)

    ps1 = PartialShape([3])
    ps2 = PartialShape.dynamic()
    assert ps1.compatible(ps2)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([4])
    assert ps1.compatible(ps2)

    ps1 = PartialShape([2, -1, 3, -1, 5])
    ps2 = PartialShape([2, -1, -1, 4, 5])
    assert ps1.compatible(ps2)

    ps1 = PartialShape([2, -1, 3, -1, 5])
    ps2 = PartialShape([1, -1, -1, 4, 5])
    assert not ps1.compatible(ps2)


def test_partial_shape_same_scheme():
    ps1 = PartialShape([1, 2, -1])
    ps2 = PartialShape([1, 3, -1])
    assert not ps1.same_scheme(ps2)

    ps1 = PartialShape([1, 2, -1])
    ps2 = PartialShape([1, 2, -1])
    assert ps1.same_scheme(ps2)

    ps1 = PartialShape([1, 2, 3])
    ps2 = PartialShape([1, 2, 3])
    assert ps1.same_scheme(ps2)

    ps1 = PartialShape([-1, 2, 3])
    ps2 = PartialShape([1, -1, 3])
    assert not ps1.same_scheme(ps2)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.same_scheme(ps2)


def test_partial_shape_refinement():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert ps2.relaxes(ps1)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([3, -1, 7, 9])
    assert not ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert not ps2.relaxes(ps1)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([3, 5, 7, 9])
    assert not ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert not ps2.relaxes(ps1)


def test_partial_shape_equals():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1 == ps2

    ps1 = PartialShape([1, 2, 3])
    ps2 = PartialShape([1, 2, 3])
    assert ps1 == ps2

    shape = Shape([1, 2, 3])
    ps = PartialShape([1, 2, 3])
    assert shape == ps
