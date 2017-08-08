// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

#include "axes.hpp"
#include "util.hpp"

using namespace ngraph;

slice::slice(int64_t start, int64_t stop, int64_t step)
    : m_start{(size_t)start}
    , m_stop{(size_t)stop}
    , m_step{step}
{
    if (step != 1 && step != -1)
    {
        throw std::invalid_argument("slice step must be 1 or -1");
    }

    if (start == -1)
    {
        m_start = 0;
    }
    if (stop == -1)
    {
        m_stop = std::numeric_limits<size_t>::max();
    }

    if (m_step > 0)
    {
        m_start = std::min<size_t>(m_start, m_stop);
    }
    else
    {
        m_start = std::max<size_t>(m_start, m_stop);
    }
}

size_t slice::sliced_length(size_t length) const
{
    size_t start = m_start;
    size_t stop  = std::min<size_t>(m_stop, length);

    size_t rc;
    if (m_step == 1)
    {
        rc = std::max<size_t>(stop - start, 0);
    }
    else if (m_step == -1)
    {
        rc = std::max<size_t>(m_start - m_stop, 0);
    }
    else
    {
        throw std::runtime_error("slice step must be 1 or -1");
    }
    return rc;
}

// def default_dtype(dtype=None):
//     if dtype is None:
//         dtype = np.dtype(np.float32)
//     elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
//         try:
//             dtype = np.dtype(dtype)
//         except TypeError:
//             raise TypeError("Could not cast {} to np.dtype".format(dtype))
//     return dtype

// def default_int_dtype(dtype=None):
//     if dtype is None:
//         dtype = np.dtype(np.int32)
//     elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
//         try:
//             dtype = np.dtype(dtype)
//         except TypeError:
//             raise TypeError("Could not cast {} to np.dtype".format(dtype))
//     return dtype

Axis ngraph::make_axis(size_t length, const std::string& name, bool batch, bool recurrent)
{
    return Axis(length, name);
}

Axes ngraph::make_axes(const std::vector<Axis>& axis_list)
{
    return Axes(axis_list);
}

//================================================================================================
// Axis
//     An Axis labels a dimension of a tensor. The op-graph uses
//     the identity of Axis objects to pair and specify dimensions in
//     symbolic expressions. This system has several advantages over
//     using the length and position of the axis as in other frameworks:
//
//     1) Convenience. The dimensions of tensors, which may be nested
//     deep in a computation graph, can be specified without having to
//     calculate their lengths.
//
//     2) Safety. Axis labels are analogous to types in general-purpose
//     programming languages, allowing objects to interact only when
//     they are permitted to do so in advance. In symbolic computation,
//     this prevents interference between axes that happen to have the
//     same lengths but are logically distinct, e.g. if the number of
//     training examples and the number of input features are both 50.
//
//     TODO: Please add to the list...
//
//     Arguments:
//         length: The length of the axis.
//         batch: Whether the axis is a batch axis.
//         recurrent: Whether the axis is a recurrent axis.
//================================================================================================
size_t Axis::__name_counter = 0;

Axis::Axis()
    : Axis(0, "")
{
}

Axis::Axis(size_t length, const std::string& new_name)
    : name{new_name}
    , uuid{uuid_type()}
    , __length{length}
{
    if (name.size() == 0)
    {
        std::stringstream ss;
        ss << "Axis_" << __name_counter++;
        name = ss.str();
    }
}

bool Axis::is_flattened() const
{
    return false;
}

bool Axis::is_batch() const
{
    return name == "N";
}

bool Axis::is_recurrent() const
{
    return name == "REC";
}

bool Axis::is_channel() const
{
    return name == "C";
}

size_t Axis::length() const
{
    return __length;
}

void Axis::length(size_t l)
{
    __length = l;
}

std::ostream& ngraph::operator<<(std::ostream& out, const Axis& axis)
{
    out << axis.to_string();
    return out;
}

std::string Axis::to_string() const
{
    std::stringstream ss;
    ss << "Axis(" << name << ": " << length() << ")";
    return ss.str();
}

bool Axis::operator==(const Axis& other) const
{
    return name == other.name;
}

bool Axis::operator!=(const Axis& other) const
{
    return !(*this == other);
}

bool Axis::operator<(const Axis& other) const
{
    bool rc;
    if (this->name == other.name)
    {
        rc = this->length() < other.length();
    }
    else
    {
        rc = this->name < other.name;
    }
    return rc;
}

// def _sliced_length(s, incoming_length):
//     start, stop, step = s.indices(incoming_length)

//     # max with 0 so we dont ever return a negative length.  This
//     # matches how python handles it internally.  Raising an exception
//     # might also be reasonable.
//     if step == 1:
//         return max(stop - start, 0)
//     elif step == -1:
//         return max(start - stop, 0)
//     else:
//         _validate_slice(s)

// // def _validate_slice(s):
// void validate_slice(const slice& s)
// {
// //     if s.step not in (-1, 1, None):
// //         raise ValueError((
// //             'SlicedAxis cant currently handle a step size other '
// //             'than -1, 1 or None.  Was given {step} in slice {slice}'
// //         ).format(
// //             step=s.step,
// //             slice=s,
// //         ))
// }

Axis ngraph::slice_axis(const Axis& axis, const slice& s)
{
    // _validate_slice(s)

    // # get sliced length
    // new_length = None if axis.length is None else _sliced_length(s, axis.length)
    auto new_length = s.sliced_length(axis.length());

    // # create sliced axis
    // new_axis = make_axis(length=new_length,
    //                      name=axis.name)
    return make_axis(new_length, axis.name);
    // return new_axis
}

// def duplicates(arr):
//     """
//     Returns a list of Axis objects which have duplicate names in arr

//     Arguments:
//         arr: The iterable of Axis objects to check for duplicates in.

//     Returns:
//         list of Axis: duplicate Axis found in arr
//     """
std::vector<std::string> ngraph::duplicates(const std::vector<Axis>& ax)
{
    std::map<std::string, size_t> counts;
    std::vector<std::string>      rc;
    for (const Axis& axis : ax)
    {
        auto it = counts.find(axis.name);
        if (it == counts.end())
        {
            counts.insert({axis.name, 1});
        }
        else
        {
            it->second++;
        }
    }
    for (auto p : counts)
    {
        if (p.second > 1)
        {
            rc.push_back(p.first);
        }
    }
    return rc;
}

// def with_args_as_axes(f):
//     """
//     A decorator to cast arguments to axes.

//     Arguments:
//         f: The function to be decorated.

//     Returns:
//         The decorated function.
//     """
//     @wraps(f)
//     def wrapper(*args):
//         """
//         The decorated function. Performs the conversion
//         to Axes.

//         Arguments:
//           *args: Arguments intended for the original function.

//         Returns:
//             Return value of the original function.
//         """
//         args = [Axes(arg) for arg in args]
//         return f(*args)
//     return wrapper

//================================================================================================
// Axes
//     An Axes is a tuple of Axis objects used as a label for a tensor's
//     dimensions.
//================================================================================================
Axes::Axes()
    : uuid{}
{
}

Axes::Axes(const Axis& axis)
{
    axes.push_back(axis);
}

Axes::Axes(const std::vector<Axis>& axis_list)
{
    axes = axis_list;
    check_duplicates();
}

Axes::Axes(const std::initializer_list<Axes>& list)
{
    axes = convert(std::vector<Axes>(list));
    check_duplicates();
}

size_t Axes::size() const
{
    return axes.size();
}

void Axes::check_duplicates()
{
    auto dups = duplicates(axes);
    if (dups.size() > 0)
    {
        std::stringstream ss;
        ss << "The axes labels of a tensor cannot contain duplicates. Found: " << join(dups, ", ");
        throw std::invalid_argument(ss.str());
    }
}

const Axis& Axes::operator[](size_t index) const
{
    return axes[index];
}

// Axis Axes::operator[](const slice&) const
// {
// }

// class Axes(object):
//     """
//     """

//     def __init__(self, axes=None):
//         if axes is None:
//             axes = []
//         elif isinstance(axes, Axis):
//             axes = [axes]
//         elif isinstance(axes, types.GeneratorType):
//             axes = tuple(axes)
//         elif isinstance(axes, (list, tuple)) and not isinstance(axes, Axes):
//             axes = tuple(axes)

//         def convert(seq):

//         axes = convert(axes)

//         for x in axes:
//             if not isinstance(x, Axis):
//                 raise ValueError((
//                     'tried to initialize an Axes with object type '
//                     '{found_type}.  all values should be an instance '
//                     'of a type which inherits from Axis.'
//                 ).format(
//                     found_type=type(x),
//                 ))

//         if duplicates(axes):
//             raise ValueError(
//                 'The axes labels of a tensor cannot contain duplicates.  Found: {}'
//                 .format(str(duplicates(axes)))
//             )
//         self._axes = tuple(axes)
//         self.uuid = uuid.uuid4()

//     @property
//     def full_lengths(self):
//         """
//         Returns all information about the lengths of the axis objects
//         in this Axes in the form of a nested tuple. An element of the
//         outer tuple that is itself a tuple contains the restored lengths
//         of axes that have been flattened in this Axis object.

//         Returns:
//             tuple: A nested tuple with the axis lengths.
//         """
//         return tuple(x.axes.full_lengths if x.is_flattened
//                      else x.length for x in self)

//     @property
//     def names(self):
//         """
//         Returns:
//             tuple: The names of the outer axes.
//         """
//         return tuple(x.name for x in self)

std::vector<Axis> Axes::convert(const Axes& ax)
{
    std::vector<Axis> rc;
    for (const Axis& axis : ax.axes)
    {
        rc.push_back(axis);
    }
    return rc;
}

std::vector<Axis> Axes::convert(const std::vector<Axes>& list)
{
    std::vector<Axis> rc;
    for (const Axes& ax : list)
    {
        if (ax.axes.size() == 1)
        {
            rc.push_back(ax.axes[0]);
        }
        else
        {
            std::vector<Axis> tmp = convert(ax);
            Axes              t1(tmp);
            auto              x = t1.flatten();
            rc.push_back(x);
        }
    }
    return rc;
}

std::vector<size_t> Axes::lengths() const
{
    // return tuple(x.length for x in self)
    std::vector<size_t> rc;
    for (auto a : axes)
    {
        rc.push_back(a.length());
    }
    return rc;
}

//     def batch_axes(self):
//         """
//         Returns:
//             The tensor's batch Axis wrapped in an Axes object if there is one
//             on this tensor, otherwise returns None
//         """
//         batch_axis = self.batch_axis()
//         if batch_axis:
//             return Axes([batch_axis])
//         else:
//             return None

//     def batch_axis(self):
//         """
//         Returns:
//             The tensor's batch Axis or None if there isn't one.
//         """
//         for axis in self:
//             if axis.is_batch:
//                 return axis

//     def channel_axis(self):
//         """
//         Returns:
//             The tensor's batch Axis or None if there isn't one.
//         """
//         for axis in self:
//             if axis.is_channel:
//                 return axis

//     def spatial_axes(self):
//         """
//         Returns:
//             The Axes subset that are not batch, recurrent, or channel axes.
//         """
//         return self.feature_axes() - self.channel_axis()

//     def sample_axes(self):
//         """
//         Returns:
//             The Axes subset that are not batch axes.
//         """
//         return Axes(axis for axis in self if not axis.is_batch)

//     def feature_axes(self):
//         """
//         Returns:
//             The Axes subset that are not batch or recurrent axes.
//         """
//         return Axes(axis for axis in self if not axis.is_batch and not axis.is_recurrent)

//     def recurrent_axis(self):
//         """
//         Returns:
//             The tensor's recurrent Axis or None if there isn't one.
//         """
//         for axis in self:
//             if axis.is_recurrent:
//                 return axis

Axis Axes::flatten(bool force) const
{
    Axis rc;
    if (!force && axes.size() == 1)
    {
        rc = axes[0];
    }
    else
    {
        rc = FlattenedAxis(axes);
    }
    return rc;
}

//     def set_shape(self, shape):
//         """
//         Set shape of Axes

//         Args:
//             shape: tuple or list of shapes, must be the same length as the axes
//         """
//         if len(shape) != len(self._axes):
//             raise ValueError("shape's length %s must be equal to axes' length"
//                              "%s" % (len(shape), len(self)))
//         for axis, length in zip(self._axes, shape):
//             axis.length = length

//     def find_by_name(self, name):
//         return Axes(axis for axis in self if axis.name == name)

//     def __iter__(self):
//         return self._axes.__iter__()

//     def __len__(self):
//         return len(self._axes)

//     def __getitem__(self, item):
//         if isinstance(item, slice):
//             return Axes(self._axes.__getitem__(item))
//         else:
//             return self._axes.__getitem__(item)

//     def __getslice__(self, i, j):
//         return self.__getitem__(slice(i, j))

Axes Axes::operator+(const Axes& other)
{
    // other = make_axes(other)
    // common_axes = self & other
    Axes common_axes = *this & other;
    if (common_axes.size() != 0)
    {
        std::stringstream ss;
        ss << "Trying to concatenate " << *this << " with " << other;
        ss << ", but they have common axes " << common_axes << ", which is not allowed.";
        throw std::invalid_argument(ss.str());
    }
    std::vector<Axis> rc = axes;
    rc.insert(rc.end(), other.axes.begin(), other.axes.end());
    return Axes(rc);
}

Axes Axes::operator-(const Axes& other)
{
    std::vector<Axis> axis_list;
    for (const Axis& axis : axes)
    {
        if (!contains(other.axes, axis))
        {
            axis_list.push_back(axis);
        }
    }
    return Axes(axis_list);
}

Axes Axes::operator|(const Axes& other)
{
    std::vector<Axis> axis_list = axes;
    for (const Axis& axis : other.axes)
    {
        if (!contains(axes, axis))
        {
            axis_list.push_back(axis);
        }
    }
    return Axes(axis_list);
}

Axes Axes::operator&(const Axes& other)
{
    std::vector<Axis> axis_list;
    for (const Axis& axis : axes)
    {
        if (contains(other.axes, axis))
        {
            axis_list.push_back(axis);
        }
    }
    return Axes(axis_list);
}

bool Axes::operator==(const Axes& other) const
{
    bool rc = axes.size() == other.size();
    if (rc)
    {
        for (int i = 0; i < axes.size(); i++)
        {
            if (axes[i] != other.axes[i])
            {
                rc = false;
                break;
            }
        }
    }
    return rc;
}

bool Axes::operator!=(const Axes& other) const
{
    return !(*this == other);
}

//     def __nonzero__(self):
//         """ Axes considered nonzero if axes are nonzero. """
//         return bool(self._axes)

//     def __hash__(self):
//         return hash(self._axes)

bool Axes::is_sub_set(const Axes& other) const
{
    bool rc = true;
    for (const Axis& axis : other.axes)
    {
        if (!contains(this->axes, axis))
        {
            rc = false;
            break;
        }
    }
    return rc;
}

bool Axes::is_super_set(const Axes& other) const
{
    bool rc = true;
    for (const Axis& axis : axes)
    {
        if (!contains(other.axes, axis))
        {
            rc = false;
            break;
        }
    }
    return rc;
}

bool Axes::is_equal_set(const Axes& other) const
{
    bool rc = axes.size() == other.axes.size();
    for (const Axis& axis : axes)
    {
        if (!contains(other.axes, axis))
        {
            rc = false;
            break;
        }
    }
    return rc;
}

bool Axes::is_not_equal_set(const Axes& other) const
{
    return !is_equal_set(other);
}

bool Axes::operator<(const Axes& other) const
{
    int rc = false;
    if (this->axes.size() == other.axes.size() && this->axes.size() > 0)
    {
        rc = this->axes[0] < other.axes[0];
    }
    else
    {
        rc = this->axes.size() < other.axes.size();
    }
    return rc;
}

//     @property
//     def T(self):
//         return Axes(axis.T for axis in self)

//     def index(self, axis):
//         """
//         Returns the index of an axis

//         Arguments:
//             axis: The axis to search for.

//         Returns:
//             The index.
//         """
//         return self._axes.index(axis)

//     @staticmethod
//     @with_args_as_axes
//     def assert_valid_broadcast(axes, new_axes):
//         """
//         Checks whether axes can be broadcasted to new_axes. We require
//         that the components of axes be laid out in the same order in new_axes.

//         Axes:
//             axes: The original axes.
//             new_axes: The broadcasted axes.

//         Returns:
//             True if axes can be broadcasted to new_axes, False otherwise.
//         """
//         removed_axes = axes - new_axes

//         if removed_axes:
//             raise ValueError(("The new_axes of a broadcast operation must "
//                               "include all of the axes from the origional set "
//                               "of axes. \n"
//                               "  original axes: {axes}\n"
//                               "  new axes: {new_axes}\n"
//                               "  missing axes: {removed_axes}").format(
//                 axes=axes,
//                 new_axes=new_axes,
//                 removed_axes=removed_axes,
//             ))

//     @staticmethod
//     @with_args_as_axes
//     def is_valid_flatten_or_unflatten(src_axes, dst_axes):
//         """
//         Checks whether we can flatten OR unflatten from src_axes to dst_axes.

//         The requirements are that the components of axes should all be
//         present in new_axes and that they should be laid out in the same
//         order. This check is symmetric.
//         """

//         # inflate
//         src_axes = Axes.as_flattened_list(src_axes)
//         dst_axes = Axes.as_flattened_list(dst_axes)

//         # check equal number of Axis
//         if len(src_axes) != len(dst_axes):
//             return False

//         # check all Axis are equal
//         equal = [src == dst for src, dst in zip(src_axes, dst_axes)]
//         return all(equal)

//     @staticmethod
//     @with_args_as_axes
//     def assert_valid_flatten(unflattend_axes, flattened_axes):
//         """
//         Checks whther axes can safely be flattened to produce new_axes.
//         The requirements are that the components of axes should all be
//         present in new_axes and that they should be laid out in the same
//         order.

//         Arguments:
//             unflattend_axes: The original axes.
//             flattened_axes: The flattened axes.

//         Returns:
//             True if axes can be safely flattened to new_axes, False otherwise.
//         """
//         if not Axes.is_valid_flatten_or_unflatten(unflattend_axes, flattened_axes):
//             raise ValueError("Trying to flatten:\n%s\nto:\n%s.\n"
//                              "But they are of different lengths, or the axes"
//                              "layouts are different"
//                              % (unflattend_axes, flattened_axes))

//     @staticmethod
//     @with_args_as_axes
//     def assert_valid_unflatten(flattened_axes, unflattend_axes):
//         """
//         Checks whether axes can safely be unflattened to produce new_axes.
//         The requirements are that the components of axes should all be
//         present in new_axes and that they should be laid out in the same
//         order.

//         Arguments:
//             flattened_axes: The original axes.
//             unflattend_axes: The unflattened axes.

//         Returns:
//             True if axes can be safely unflattened to new_axes, False otherwise.
//         """
//         if not Axes.is_valid_flatten_or_unflatten(flattened_axes, unflattend_axes):
//             raise ValueError("Trying to unflatten:\n%s\nto:\n%s.\n"
//                              "But they are of different lengths, or the axes"
//                              "layouts are different"
//                              % (unflattend_axes, flattened_axes))

//     @property
//     def size(self):
//         """
//         TODO: delete this method, the size should come from the tensor
//         """
//         return int(np.prod(self.lengths))

std::ostream& ngraph::operator<<(std::ostream& out, const Axes& axes)
{
    out << "Axes(";
    out << join(axes.axes, ", ");
    out << ")";
    return out;
}
//     def __repr__(self):
//         return 'Axes({})'.format(
//             ', '.join(map(repr, self))
//         )

//     def __str__(self):
//         return ', '.join(map(str, self))

//     @staticmethod
//     def as_nested_list(axes):
//         """
//         Converts Axes to a list of axes with flattened axes expressed as nested lists

//         Returns:
//             Nested list of Axis objects
//         """
//         if isinstance(axes, (Axes, list)):
//             return [Axes.as_nested_list(a) for a in axes]
//         elif isinstance(axes, FlattenedAxis):
//             return [Axes.as_nested_list(a) for a in axes.axes]
//         elif isinstance(axes, Axis):
//             return axes

//     @staticmethod
//     def as_flattened_list(axes):
//         """
//         Converts Axes to a list of axes with flattened axes expanded recursively.

//         Returns:
//             List of Axis objects
//         """
//         axes_list = [list(axis.axes) if axis.is_flattened else [axis]
//                      for axis in axes]
//         axes = list(itertools.chain.from_iterable(axes_list))

//         # inflate recursively
//         if any([axis.is_flattened for axis in axes]):
//             return Axes.as_flattened_list(axes)
//         else:
//             return axes

//================================================================================================
// DuplicateAxisNames
//================================================================================================

// class DuplicateAxisNames(ValueError):
//     def __init__(self, message, duplicate_axis_names):
//         super(DuplicateAxisNames, self).__init__(message)

//         self.duplicate_axis_names = duplicate_axis_names

//================================================================================================
// IncompatibleAxesError
//================================================================================================

// class IncompatibleAxesError(ValueError):
//     pass

//================================================================================================
// UnmatchedAxesError
//================================================================================================

// class UnmatchedAxesError(IncompatibleAxesError):
//     pass

//================================================================================================
// AxesMap
//     AxesMap provides a way to define a axis name mapping: {Axis.name: Axis.name} and
//     then apply this mapping to an Axes and get new Axes out.
//
//     Right now AxesMap is implemented as immutible because I didn't want to deal with
//     enforcing _assert_valid_axes_map on every method which mutates a dict and I didn't
//     need a mutable datastructure anyway.  Feel free to make it mutable and add in
//     invariant enforcement.
//================================================================================================
AxesMap::AxesMap(const std::pair<std::string, std::string>& p)
{
    this->insert(p);
}

AxesMap::AxesMap(std::initializer_list<std::pair<std::string, std::string>> list)
{
    this->insert(list.begin(), list.end());
    assert_valid_axes_map();
}

//     def __init__(self, *args, **kwargs):
//         def replace_axis_with_name(x):
//             if isinstance(x, Axis):
//                 return x.name
//             return x

//         # strip axis objects into just names
//         super(AxesMap, self).__init__({
//             replace_axis_with_name(k): replace_axis_with_name(v)
//             for k, v in dict(*args, **kwargs).items()
//         })

//         self._assert_valid_axes_map()

Axes AxesMap::map_axes(const Axes& ax) const
{
    std::vector<Axis> mapped_list;
    for (const Axis& axis : ax)
    {
        mapped_list.push_back(map_axis(axis));
    }
    return make_axes(mapped_list);
}

Axis AxesMap::map_axis(const Axis& old_axis) const
{
    Axis rc = old_axis;
    if (contains_key(*this, old_axis.name))
    {
        rc = make_axis(old_axis.length(), this->at(old_axis.name));
    }
    return rc;
}

std::map<std::string, std::set<std::string>> AxesMap::duplicate_axis_names()
{
    std::map<std::string, std::set<std::string>> counts;
    for (auto p : *this)
    {
        counts[p.second].insert(p.first);
    }
    std::map<std::string, std::set<std::string>> rc;
    for (auto p : counts)
    {
        if (p.second.size() > 1)
        {
            rc.insert(p);
        }
    }
    return rc;
}

void AxesMap::assert_valid_axes_map()
{
    auto duplicate_names = duplicate_axis_names();

    // if there are duplicate_axis_names throw an exception
    if (duplicate_names.size() > 0)
    {
        std::stringstream ss;
        ss << "AxesMap can not have duplicate names, but found:";
        for (auto p : duplicate_names)
        {
            ss << "\n    " << p.first << " maps to " << join(p.second, ", ");
        }

        throw std::invalid_argument(ss.str());
    }
}

//     def invert(self):
//         return {v: k for k, v in self.items()}

// def _reduce_nested(elem, agg, func):
//     """
//     Reduces a nested sequence by applying a function to each
//     of its elements and returns an aggregation.

//     Arguments:
//       elem: The object to be reduced, either a sequence
//         or a singleton.
//       agg: A variable holding information collected
//         as the sequence is collapsed.
//       func: A function to augment the aggregate by processing
//         a singleton. Should have the form func(agg, elem) -> agg

//     Returns:
//         agg: The final aggregate returned by the function.
//     """
//     if isinstance(elem, collections.Iterable):
//         for sub in elem:
//             agg = _reduce_nested(sub, agg, func)
//         return agg
//     else:
//         return func(agg, elem)

//================================================================================================
// FlattenedAxis
//     A FlattenedAxis has length which is the product of the lengths of all
//     Axis in the axes.  The original Axes object is stored so that we can later
//     unflatten this Axis back to its original component Axis.
//
//     Notes: since we allows Axis to have duplicated names globally, NameableValue
//     is not used here.
//================================================================================================
FlattenedAxis::FlattenedAxis(const std::vector<Axis>& list, const std::string& new_name)
{
    // get length
    Axes ax(list);
    // if len(axes) == 1 and axes[0].is_flattened:
    //     pass
    // length = reduce(operator.mul, axes.lengths, 1)
    auto lengths = ax.lengths();
    __length     = ngraph::reduce(lengths.begin(), lengths.end(), ngraph::mul<size_t>);

    // # set name
    // name = '%s_%s' % (type(self).__name__, type(self).__name_counter)
    // type(self).__name_counter += 1
    name = new_name;
    if (name.size() == 0)
    {
        std::stringstream ss;
        ss << "Axis_" << __name_counter++;
        name = ss.str();
    }

    // # parent constructor
    // super(FlattenedAxis, self).__init__(length=length, name=name, **kwargs)
    // self._axes = axes
    axes = list;
}

std::ostream& ngraph::operator<<(std::ostream& out, const FlattenedAxis& obj)
{
    out << obj.to_string();
    return out;
}

std::string FlattenedAxis::to_string() const
{
    std::stringstream ss;
    ss << "FlattenedAxis(" << join(axes, ", ") << ")";
    return ss.str();
}

// def _make_stride(inner_size, axis, fsz):
//     """
//     Generates a nested tuple that provides the striding information
//     for an occurrence of axis. If the axis is a FlattenedAxis, the
//     stride will be a tuple containing the strides of each collapsed
//     axis. Otherwise, the stride will be an integer.

//     Arguments:
//         inner_size: The total size of all dimensions smaller than this
//         axis, i.e. all axes to the right of this one when they are
//         laid out in c-contiguous order.
//         axis: The axis for which we are generating a stride.
//         fsz: A nested tuple supplying the sizes of each dimension collapsed
//         into the axis. The size may be larger than the length of the axis.

//     Returns:
//         inner_size: The total size of this axis and all smaller dimensions.
//         stride: The stride given to the axis.
//     """
//     if axis.is_flattened:
//         return _make_strides(inner_size, axis.axes, fsz)
//     else:
//         stride = inner_size
//         inner_size *= fsz
//         return inner_size, stride

// def _make_strides(inner_size, axes, full_sizes):
//     """
//     Generates a tuple of strides for a set of axes. See _make_stride
//     for a description of the stride given to each axis.

//     Arguments:
//         inner_size: The total size of all dimensions smaller than
//         the axes.
//         axes: The axes for which we are generating strides.
//         full_sizes: The size of each axis.

//     Returns:
//         inner_size: The total size of these axes and all smaller dimensions.
//         strides: The strides generated for the axes.
//     """
//     full_strides = []
//     for axis, fsz in reversed(list(zip(axes, full_sizes))):
//         inner_size, stride = _make_stride(inner_size, axis, fsz)
//         full_strides.append(stride)
//     return inner_size, tuple(reversed(full_strides))

//================================================================================================
// TensorDescription
//     Description of a tensor that will be allocated in hardware.
//
//     Names the tensor's dimensions with axes and holds pointers to the
//     buffer allocated by the analysis and the backend tensor value
//     (e.g. a cpu or gpu tensor).
//
//     Arguments:
//         axes: Axes of the tensor.
//         base: If a view, the viewed tensor's description.
//         dtype: The type of the tensor.
//         full_strides: The strides of each axis.
//         full_sizes: The allocated size of each axis (may be larger than the axis).
//         offset: An offset into the viewed tensor.
//         next_tensor_decription: In a reshape, tensor description of reshaped tensor.
//         is_persistent: The tensor should be persistent, i.e. survive from computation to
//             computation.
//         is_input: The device tensor can be written from the host.
//         **kwargs: Additional args for related classes.
//================================================================================================
TensorDescription::TensorDescription(op_ptr                 _op,
                                     const Axes&            _axes,
                                     tensor_description_ptr base,
                                     //   layout,
                                     ElementType           et,
                                     ngraph::tensor_stride _full_strides,
                                     ngraph::tensor_size   _full_sizes,
                                     size_t                _offset,
                                     TensorDescription*    next_tensor_decription,
                                     const std::string&    _name,
                                     bool                  is_persistent,
                                     bool                  is_input,
                                     bool                  is_placeholder)
    : NameableValue(_name)
    , op{_op}
    , axes{_axes}
    , __is_persistent{is_persistent}
    , __is_input{is_input}
    , __is_placeholder{is_placeholder}
    , __base{base}
    , dtype{et}
    , full_sizes{_full_sizes}
    , full_strides{_full_strides}
{
    // super(TensorDescription, self).__init__(**kwargs)
    // # TODO: get the default type from the backend. May not always be numpy.
    // # TODO: support flattening, unflattening, other complex reshapes
    // axes = Axes(axes)
    // self.axes = axes
    // self.__layout = layout
    // self.__value = None
    // self.__buffer = None
    // self.__register = None
    // self.__base = base
    // self.dtype = default_dtype(dtype)
    // self.offset = offset
    // self.ndim = len(self.axes)
    // self.full_sizes = tuple(full_sizes) if full_sizes is not None \
    //     else self.axes.full_lengths
    // self.next_tensor_description = next_tensor_description
    // self.__is_persistent = is_persistent
    // self.__is_input = is_input
    // self.__is_placeholder = is_placeholder
    // self.op = op
    // if not isinstance(self.name, str):
    //     raise ValueError()
    // for axis in axes:
    //     if axis.length is None:
    //         raise ValueError((
    //             'axes used in the constructor of TensorDescription must '
    //             'always have non-None length.  Axis {axis} has length '
    //             'None.'
    //         ).format(axis=axis))

    // if full_strides is None:
    //     _, full_strides = _make_strides(
    //         self.dtype.itemsize,
    //         self.axes,
    //         self.full_sizes
    //     )
    //     self.full_strides = full_strides
    // else:
    //     self.full_strides = tuple(full_strides)

    // assert len(self.full_sizes) == self.ndim, \
    //     "Sizes must have same number of dimensions as axes"
    // assert len(self.full_strides) == self.ndim, \
    //     "Strides must have same number of dimensions as axes"
}

//     def __repr__(self):
//         return self.base.name

ElementType TensorDescription::element_type() const
{
    return dtype;
}

bool TensorDescription::is_persistent() const
{
    return __is_persistent;
}

bool TensorDescription::is_input() const
{
    return __is_input;
}

bool TensorDescription::is_placeholder() const
{
    return __is_placeholder;
}

//     @property
//     def parameter_key(self):
//         """
//         Returns: A tuple that can be used to tell if two views of a tensor are equivalent.
//         """
//         return (self.shape, self.dtype, self.offset, self.strides, self.layout)

//     @property
axes_key_t TensorDescription::axes_key() const
{
    std::hash<Axes>   axes_hash;
    std::hash<size_t> offset_hash;
    // std::hash<decltype(strides)> strides_hash;
    // std::hash<decltype(layout)> layout_hash;

    std::vector<size_t> hash_list;
    hash_list.push_back(axes_hash(axes));
    // hash_list.push_back(hash_combine(shape));
    hash_list.push_back(dtype.hash());
    hash_list.push_back(offset_hash(offset));

    // TODO: add strides and layout to hash

    //     def axes_key(self):
    //         return (self.axes, self.shape, self.dtype, self.offset, self.strides, self.layout)

    return hash_combine(hash_list);
};

//     def flatten(self, new_axes):
//         """
//         Flattens a tensor description to give it the Axes in new_axes.
//         See Axes.assert_valid_flatten for a description of permitted values of new_axes.

//         Arguments:
//             new_axes: The Axes of the flattened tensor description.

//         Returns:
//             The reshaped tensor description.
//         """
//         new_axes = Axes(new_axes)
//         Axes.assert_valid_flatten(self.axes, new_axes)

//         new_strides = []
//         new_sizes = []
//         idx = 0
//         for new_axis in new_axes:
//             if new_axis == self.axes[idx]:
//                 new_stride = self.full_strides[idx]
//                 new_size = self.full_sizes[idx]
//                 idx += 1
//             else:
//                 l = len(new_axis.axes)
//                 new_stride = self.full_strides[idx:idx + l]
//                 new_size = self.full_sizes[idx:idx + l]
//                 idx += l

//             new_strides.append(new_stride)
//             new_sizes.append(new_size)

//         return TensorDescription(
//             new_axes,
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=new_strides,
//             full_sizes=new_sizes,
//             offset=self.offset,
//             next_tensor_description=self,
//             name=self.name + 'rFlatten',
//         )

//     def unflatten(self, new_axes):
//         """
//         Unflattens a tensor description to give it the Axes in new_axes.
//         See Axes.assert_valid_unflatten for a description of the permitted values of
//         new_axes

//         Arguments:
//             new_axes: The Axes of the unflattened TensorDescription.

//         Returns:
//             The unflattened tensor description.
//         """
//         def find_axis_stride_and_length(axis):
//             """
//             Find the stride and length for an axis.

//             Start at the current tensor description and then work back
//             through reshapings of it looking for a mention of the axis
//             that can be used to determine the storage stride and offset.

//             Args:
//                 axis: The axis.

//             Returns:
//                 stride, length of axis

//             """
//             td = self
//             while td is not None:
//                 for idx, a in enumerate(td.axes):
//                     # Try to find a match for axis in this td
//                     full_strides = td.full_strides[idx]
//                     full_sizes = td.full_sizes[idx]
//                     if a == axis:
//                         return full_strides, full_sizes

//                     if a.is_flattened:
//                         # Can be embedded ina a flattened axis description
//                         if not isinstance(full_strides, tuple):
//                             # An axis cast can lose striding info, so need to
//                             # recreate it from the axis lengths. Being flattened
//                             # implies C-contiguous
//                             stride = full_strides
//                             full_strides = []
//                             full_sizes = []
//                             for s in reversed(a.axes):
//                                 full_sizes.insert(0, s.length)
//                                 full_strides.insert(0, stride)
//                                 stride = stride * s.length

//                         # Now search for axis in the flattened axis
//                         for sub_idx, b in enumerate(a.axes):
//                             if b == axis:
//                                 return full_strides[sub_idx], full_sizes[sub_idx]

//                 # Move on to the next tensor description in the reshaping chain
//                 td = td.next_tensor_description

//             # Sometimes we just don't have enough information.
//             raise ValueError()

//         new_axes = Axes(new_axes)
//         Axes.assert_valid_unflatten(self.axes, new_axes)

//         new_strides = []
//         new_sizes = []
//         for new_axis in new_axes:
//             stride, size = find_axis_stride_and_length(new_axis)
//             new_strides.append(stride)
//             new_sizes.append(size)

//         return TensorDescription(
//             new_axes,
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=new_strides,
//             full_sizes=new_sizes,
//             offset=self.offset,
//             next_tensor_description=self,
//             name=self.name + 'rUnflatten',
//         )

//     def transpose(self):
//         """
//         Reverses the axes of the tensor description.

//         Retuns:
//             A tensor description with the axes reversed.
//         """
//         new_axes = reversed(self.axes)
//         full_sizes = reversed(self.full_sizes)
//         full_strides = reversed(self.full_strides)
//         return TensorDescription(
//             Axes(new_axes),
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=tuple(full_strides),
//             full_sizes=tuple(full_sizes),
//             offset=self.offset,
//             next_tensor_description=self,
//             name=self.name + 'rTranspose',
//         )

//     def clone(self):
//         """
//         Creates a copy of this tensor description

//         Retuns:
//             A copy of this tensor description
//         """
//         return TensorDescription(
//             self.axes,
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=self.full_strides,
//             full_sizes=self.full_sizes,
//             offset=self.offset,
//             next_tensor_description=self.next_tensor_description,
//             name=self.name + 'cView',
//         )

// TensorDescription TensorDescription::broadcast(const Axes& new_axes)
// {
//     Axes::assert_valid_broadcast(axes, new_axes);
//     return reorder_and_broadcast(new_axes);
// }

//         Axes.assert_valid_broadcast(self.axes, new_axes)
//         return self.reorder_and_broadcast(new_axes)

//     def reorder(self, new_axes):
//         """
//         Shuffles axes of a tensor to give it a new shape. The axes of
//         this tensor description and new_axes must have the same elements.

//         Arguments:
//             new_axes: The axes of the reordered tensor.

//         Returns:
//             TensorDescription: The reordered tensor description.
//         """
//         if not self.axes.is_equal_set(new_axes):
//             raise ValueError((
//                 "Reorder can't change which axes are available, only the "
//                 "order.  {} and {} are different sets, not just order."
//             ).format(self, new_axes))

//         return self.reorder_and_broadcast(new_axes)

//     def reorder_and_broadcast(self, new_axes):
//         """
//         Adds or shuffles axes to give a tensor description a new shape.
//         This function is used to implement broadcast and reorder.

//         Arguments:
//             new_axes: The axes of the broadcasted or reordered tensor.

//         Returns:
//             TensorDescription: A description of the tensor after the
//             transformation.
//         """

// def zero_in_shape(tup):
// zero_in_shape()
// {
// //     if isinstance(tup, collections.Iterable):
// //         return tuple(
// //             zero_in_shape(t) for t in tup
// //         )
// //     else:
// //         return 0
// }

// TensorDescription TensorDescription::reorder_and_broadcast(const Axes& _new_axes)
// {
//     // new_axes = Axes(new_axes)
//     auto new_axes = Axes(_new_axes);
//     // new_strides = []
//     std::vector<size_t> new_strides;
//     // new_sizes = []
//     std::vector<size_t> new_sizes;
//     // for axis in new_axes:
//     for (const Axis& axis : new_axes)
//     {
//         // if axis in self.axes:
//         if (contains(axes, axis))
//         {
//             // idx = self.axes.index(axis)
//             auto idx = axes.index(axis);
//             // new_strides.append(self.full_strides[idx])
//             new_strides.push_back(full_strides[idx]);
//             // new_sizes.append(self.full_sizes[idx])
//             new_sizes.push_back(full_sizes[idx]);
//         }
//         // elif axis.is_flattened:
//         else if (axis.is_flattened())
//         {
//             // lengths = axis.axes.full_lengths
//             auto lengths = axis.axes().full_lengths();
//             // new_strides.append(zero_in_shape(lengths))
//             new_strides.push_back(zero_in_shape(lengths));
//             // new_sizes.append(lengths)
//             new_sizes.push_back(lengths);
//         }
//         // else:
//         else
//         {
//             // new_strides.append(0)
//             new_strides.push_back(0);
//             // new_sizes.append(axis.length)
//             new_sizes.push_back(axis.length());
//         }
//     }

//     return TensorDescription(
//         nullptr,
//         new_axes,
//         base,
//         dtype,
//         new_strides,
//         new_sizes,
//         offset,
//         this,
//         name() + "rReorderBroadcast"
//     );
// }

//     def cast(self, new_axes):
//         """
//         Return a tensor desciption for a view of the tensor.

//         Arguments:
//             new_axes: The axes for the view.

//         Returns:
//             The tensor description.

//         """
//         full_strides = self.full_strides
//         full_sizes = self.full_sizes
//         if self.ndim == 0:
//             full_strides = (0,) * len(new_axes)
//             full_sizes = new_axes.full_lengths

//         return TensorDescription(
//             new_axes,
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=full_strides,
//             full_sizes=full_sizes,
//             offset=self.offset,
//             next_tensor_description=self,
//             name=self.name + 'rCast',
//         )

//     def slice(self, slices, new_axes):
//         """
//         Return a tensor description for a slice view of this tensor.

//         Arguments:
//             slices: The slices to take from the tensor, each of which is
//             either an integer or a python slice. If the input has too few
//             axes for the tensor, we assume that the entire axis should be
//             taken for dimensions towards the end of the tensor.
//             new_axes: the axes to use as labels for the sliced tensor.

//         Returns:
//             The tensor description for the slice.
//         """
//         slices = list(slices)
//         while len(slices) < self.ndim:
//             slices.append(slice(None))

//         offset = self.offset
//         full_strides = []
//         full_sizes = []
//         new_index = 0

//         # check new_axes for the correct length
//         num_dimensions_out = len([s for s in slices if isinstance(s, slice)])
//         if len(new_axes) != num_dimensions_out:
//             raise ValueError((
//                 'in a slice operation, the number of axes passed in to '
//                 'new_axes ({num_new_axes}) must be the same as the number of '
//                 'slice objects in slices ({num_slices}).'
//             ).format(
//                 num_new_axes=len(new_axes),
//                 num_slices=num_dimensions_out,
//             ))

//         for s, axis, stride, size in zip(slices, self.axes, self.strides, self.sizes):
//             if isinstance(s, slice):
//                 # only increment new_axis when the input slice is a slice and
//                 # not a integer
//                 new_axis = new_axes[new_index]
//                 new_index += 1

//                 # ensure slice is of the kind we support
//                 _validate_slice(s)

//                 # ensure new_axis has the correct length
//                 new_axis.length = _sliced_length(s, axis.length)

//                 start, stop, step = s.indices(axis.length)

//                 full_strides.append(stride * step)
//                 full_sizes.append(size)

//                 idx = start
//             else:
//                 # this is a simple integer slice, ex: y = x[1]
//                 idx = s

//             # TODO: write a test that fails if abs() is removed
//             offset += idx * abs(stride)

//         return TensorDescription(
//             new_axes,
//             base=self.base,
//             dtype=self.dtype,
//             full_strides=tuple(full_strides),
//             full_sizes=tuple(full_sizes),
//             offset=offset,
//             next_tensor_description=self,
//             name=self.name + "rSlice",
//         )

std::vector<size_t> TensorDescription::shape() const
{
    return axes.lengths();
}

//     @property
//     def strides(self):
//         """The strides of the tensor."""
//         return reduce_strides(self.full_strides)

//     @property
//     def sizes(self):
//         """The allocated sizes for each axis."""
//         return tuple(_reduce_nested(_, 1, operator.mul)
//                      for _ in self.full_sizes)

size_t TensorDescription::tensor_size() const
{
    throw std::runtime_error("unimplemented");
}
//         result = self.dtype.itemsize
//         for s in self.sizes:
//             result = result * s
//         return result

//     @property
//     def c_contiguous(self):
//         """

//         Returns:
//             True if the tensor's strides are row-major contiguous.
//         """
//         s = self.dtype.itemsize
//         cstrides = []
//         for _ in reversed(self.shape):
//             cstrides.insert(0, s)
//             s = s * _
//         return tuple(cstrides) == self.strides

//     @property
//     def broadcast_contiguous(self):
//         """
//         Returns:
//             True if tensor's strides are contiguous or broadcasted
//         """
//         if self.shape == ():
//             return True

//         broadcast_axes = np.where(np.equal(self.strides, 0))[0]
//         aug_shape = list(self.shape)
//         for bcast_axis in broadcast_axes:
//             aug_shape[bcast_axis] = 1

//         s = self.dtype.itemsize
//         cstrides = []
//         for _ in reversed(aug_shape):
//             cstrides.insert(0, s)
//             s = s * _

//         for bcast_axis in broadcast_axes:
//             cstrides[bcast_axis] = 0
//         return tuple(cstrides) == self.strides

tensor_description_ptr TensorDescription::base() const
{
    return __base;
}

//     @property
//     def layout(self):
//         """The layout of the underlying storage."""
//         return self.__layout

//     @layout.setter
//     def layout(self, value):
//         """
//         Sets the backend-specific memory layout to be used by the tensor.

//         Arguments:
//           value: the layout to use

//         Returns:
//         """
//         self.__layout = value

//     @property
//     def register(self):
//         return self.base.__register

//     @register.setter
//     def register(self, value):
//         self.base.__register = value

//     def is_base(self):
//         """This tensor provides its own storage."""
//         return self.__base is None
