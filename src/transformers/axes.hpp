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

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <initializer_list>
#include <set>

#include "uuid.hpp"
#include "element_type.hpp"
#include "names.hpp"
#include "util.hpp"
#include "strides.hpp"
#include "uuid.hpp"

namespace ngraph
{
class Axes;
class Axis;
class FlattenedAxis;
class TensorDescription;
class Op;

using op_ptr                 = std::shared_ptr<Op>;
using tensor_description_ptr = std::shared_ptr<TensorDescription>;
using axes_key_t             = size_t;

class slice
{
public:
    slice(int64_t start = -1, int64_t stop = -1, int64_t step = 1);

    size_t sliced_length(size_t length) const;

private:
    size_t  m_start;
    size_t  m_stop;
    int64_t m_step;
};

//-----------------------------------------------------------------------------------------------
// default_dtype
//-----------------------------------------------------------------------------------------------
// def default_dtype(dtype=None):
//     if dtype is None:
//         dtype = np.dtype(np.float32)
//     elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
//         try:
//             dtype = np.dtype(dtype)
//         except TypeError:
//             raise TypeError("Could not cast {} to np.dtype".format(dtype))
//     return dtype

//-----------------------------------------------------------------------------------------------
// default_int_dtype
//-----------------------------------------------------------------------------------------------
// def default_int_dtype(dtype=None):
//     if dtype is None:
//         dtype = np.dtype(np.int32)
//     elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
//         try:
//             dtype = np.dtype(dtype)
//         except TypeError:
//             raise TypeError("Could not cast {} to np.dtype".format(dtype))
//     return dtype

//================================================================================================
// make_axis
//     Returns a new Axis.
//
//     Args:
//         length (int, optional): Length of the axis.
//         name (String, optional): Name of the axis.
//         batch (bool, optional): This is a batch axis. Defaults to False.
//         recurrent (bool, optional): This is a recurrent axis. Defaults to False.
//         docstring (String, optional): A docstring for the axis.
//
//     Returns:
//         Axis: A new Axis.
//================================================================================================
Axis make_axis(size_t             length,
               const std::string& name      = "",
               bool               batch     = false,
               bool               recurrent = false);

//================================================================================================
// make_axes
//     Makes an Axes object.
//
//     Args:
//         axes: A list of Axis.
//
//     Returns:
//         Axes: An Axes.
//================================================================================================
Axes make_axes(const std::vector<Axis>&);

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
class Axis
{
public:
    Axis& operator+(const Axis& rhs);
    Axis& operator-(const Axis& rhs);

    Axis();
    Axis(size_t length, const std::string& new_name);

    virtual ~Axis() {}

    void named(const std::string& new_name);

    //!-----------------------------------------------------------------------------------
    //! is_flattened
    //!     Returns:
    //!     true if this is a flattened axis.
    //!-----------------------------------------------------------------------------------
    bool is_flattened() const;

    //!-----------------------------------------------------------------------------------
    //! is_batch
    //!     Tests if an axis is a batch axis.
    //!
    //!     Returns:
    //!         bool: True if the axis is a batch axis.
    //!-----------------------------------------------------------------------------------
    bool is_batch() const;

    //!-----------------------------------------------------------------------------------
    //! is_recurrent
    //!     Tests if an axis is a recurrent axis.
    //!
    //!     Returns:
    //!         bool: True if the axis is a recurrent axis.
    //!-----------------------------------------------------------------------------------
    bool is_recurrent() const;

    //!-----------------------------------------------------------------------------------
    //! is_channel
    //!     Tests if an axis is a channel axis.
    //!
    //!     Returns:
    //!         bool: True if the axis is a channel axis.
    //!-----------------------------------------------------------------------------------
    bool is_channel() const;

    //!-----------------------------------------------------------------------------------
    //! length
    //!     Returns:
    //!         The length of the axis.
    //!-----------------------------------------------------------------------------------
    size_t length() const;

    //!-----------------------------------------------------------------------------------
    //! length
    //!-----------------------------------------------------------------------------------
    void length(size_t new_length);

    //!-----------------------------------------------------------------------------------
    //! axes
    //!-----------------------------------------------------------------------------------
    Axes axes() const;

    //!-----------------------------------------------------------------------------------
    //! operator<<
    //!-----------------------------------------------------------------------------------
    friend std::ostream& operator<<(std::ostream&, const Axis&);
    virtual std::string  to_string() const;

    //!-----------------------------------------------------------------------------------
    //! ???
    //!-----------------------------------------------------------------------------------
    //     def __str__(self):
    //         return '{name}: {length}'.format(name=self.name, length=self.length)

    //!-----------------------------------------------------------------------------------
    //! operator==
    //!-----------------------------------------------------------------------------------
    bool operator==(const Axis&) const;

    //!-----------------------------------------------------------------------------------
    //! operator==
    //!-----------------------------------------------------------------------------------
    bool operator!=(const Axis&) const;

    bool operator<(const Axis&) const;

    //!-----------------------------------------------------------------------------------
    //! hash
    //!-----------------------------------------------------------------------------------
    size_t hash() const;

    std::string   name;
    uuid_type     uuid;
    size_t        __length;
    static size_t __name_counter;
};

//-----------------------------------------------------------------------------------------------
// _sliced_length
//-----------------------------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------------------------
// _validate_slice
//-----------------------------------------------------------------------------------------------
// def _validate_slice(s):
//     if s.step not in (-1, 1, None):
//         raise ValueError((
//             'SlicedAxis cant currently handle a step size other '
//             'than -1, 1 or None.  Was given {step} in slice {slice}'
//         ).format(
//             step=s.step,
//             slice=s,
//         ))

//-----------------------------------------------------------------------------------------------
// slice_axis
//     Slice an axis, return complete new axis
//     TODO: deprecate this after the axis refactoring
//
//     Arguments:
//         axis: the axis to be sliced
//         s: slice
//
//     Returns:
//         Axis instance, the new sliced axis
//-----------------------------------------------------------------------------------------------
// def slice_axis(axis, s):
Axis slice_axis(const Axis& axis, const slice& s);

//-----------------------------------------------------------------------------------------------
// duplicates
//     Returns a list of Axis objects which have duplicate names in arr
//
//     Arguments:
//         arr: The iterable of Axis objects to check for duplicates in.
//
//     Returns:
//         list of Axis: duplicate Axis found in arr
//-----------------------------------------------------------------------------------------------
std::vector<std::string> duplicates(const std::vector<Axis>& ax);

//-----------------------------------------------------------------------------------------------
// with_args_as_axes
//     A decorator to cast arguments to axes.
//
//     Arguments:
//         f: The function to be decorated.
//
//     Returns:
//         The decorated function.
//-----------------------------------------------------------------------------------------------
// def with_args_as_axes(f):
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
class Axes
{
public:
    std::vector<Axis> axes;
    uuid_type         uuid;

    Axes();
    Axes(const Axis&);
    Axes(const std::vector<Axis>&);
    Axes(const std::initializer_list<Axes>&);

    //!-----------------------------------------------------------------------------------
    //! full_lengths
    //!    Returns all information about the lengths of the axis objects
    //!    in this Axes in the form of a nested tuple. An element of the
    //!    outer tuple that is itself a tuple contains the restored lengths
    //!    of axes that have been flattened in this Axis object.
    //!
    //!    Returns:
    //!        tuple: A nested tuple with the axis lengths.
    //!-----------------------------------------------------------------------------------
    std::vector<std::vector<size_t>> full_lengths() const;

    //!-----------------------------------------------------------------------------------
    //! names
    //!    Returns:
    //!        tuple: The names of the outer axes.
    //!-----------------------------------------------------------------------------------
    std::vector<std::string> names() const;

    //!-----------------------------------------------------------------------------------
    //! lengths
    //!    Returns:
    //!        tuple: The lengths of the outer axes.
    //!-----------------------------------------------------------------------------------
    std::vector<size_t> lengths() const;

    //!-----------------------------------------------------------------------------------
    //! batch_axes
    //!    Returns:
    //!        The tensor's batch Axis wrapped in an Axes object if there is one
    //!        on this tensor, otherwise returns None
    //!-----------------------------------------------------------------------------------
    Axes batch_axes();

    //!-----------------------------------------------------------------------------------
    //! batch_axis
    //!    Returns:
    //!        The tensor's batch Axis or None if there isn't one.
    //!-----------------------------------------------------------------------------------
    Axis batch_axis();

    //!-----------------------------------------------------------------------------------
    //! channel_axis
    //!    Returns:
    //!        The tensor's batch Axis or None if there isn't one.
    //!-----------------------------------------------------------------------------------
    Axes channel_axis();

    //!-----------------------------------------------------------------------------------
    //! spatial_axes
    //!    Returns:
    //!        The Axes subset that are not batch, recurrent, or channel axes.
    //!-----------------------------------------------------------------------------------
    Axes spatial_axes();

    //!-----------------------------------------------------------------------------------
    //! sample_axes
    //!    Returns:
    //!        The Axes subset that are not batch axes.
    //!-----------------------------------------------------------------------------------
    Axes sample_axes();

    //!-----------------------------------------------------------------------------------
    //! feature_axes
    //!    Returns:
    //!        The Axes subset that are not batch or recurrent axes.
    //!-----------------------------------------------------------------------------------
    Axes feature_axes();

    //!-----------------------------------------------------------------------------------
    //! recurrent_axis
    //!    Returns:
    //!        The tensor's recurrent Axis or None if there isn't one.
    //!-----------------------------------------------------------------------------------
    Axis recurrent_axis() const;

    //!-----------------------------------------------------------------------------------
    //! flatten
    //!    Produces flattened form of axes
    //!
    //!    Args:
    //!        force: Add a FlattenedAxis even when the axis is already flat. This is needed
    //!         when the flatten is balanced by a later unflatten, as in dot.
    //!
    //!    Returns:
    //!        A flat axis.
    //!-----------------------------------------------------------------------------------
    Axis flatten(bool force = false) const;

    //!-----------------------------------------------------------------------------------
    //! set_shape
    //!    Set shape of Axes
    //!
    //!    Args:
    //!        shape: tuple or list of shapes, must be the same length as the axes
    //!-----------------------------------------------------------------------------------
    void set_shape(std::vector<size_t> shapes);

    //!-----------------------------------------------------------------------------------
    //! find_by_name
    //!-----------------------------------------------------------------------------------
    Axes find_by_name(const std::string&);

    decltype(axes)::iterator       begin() { return axes.begin(); }
    decltype(axes)::iterator       end() { return axes.end(); }
    decltype(axes)::const_iterator begin() const { return axes.begin(); }
    decltype(axes)::const_iterator end() const { return axes.end(); }
    //     def __iter__(self):
    //         return self._axes.__iter__()

    //     def __len__(self):
    //         return len(self._axes)

    const Axis& operator[](size_t) const;
    const Axis& operator[](const slice&) const;
    //     def __getitem__(self, item):
    //         if isinstance(item, slice):
    //             return Axes(self._axes.__getitem__(item))
    //         else:
    //             return self._axes.__getitem__(item)

    //     def __getslice__(self, i, j):
    //         return self.__getitem__(slice(i, j))

    //!-----------------------------------------------------------------------------------
    //! operator+
    //!    Returns list concatenated axes. Throws exception when there are Axis
    //!    duplication.
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        current axes concatenated with the other axes
    //!-----------------------------------------------------------------------------------
    Axes operator+(const Axes&);

    //!-----------------------------------------------------------------------------------
    //! operator-
    //!    Returns ordered set difference of axes.
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        The ordered set difference of axes
    //!-----------------------------------------------------------------------------------
    Axes operator-(const Axes&);

    //!-----------------------------------------------------------------------------------
    //! operator|
    //!    Returns ordered set union of axes.
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        The ordered set union of axes
    //!-----------------------------------------------------------------------------------
    Axes operator|(const Axes&);

    //!-----------------------------------------------------------------------------------
    //! operator&
    //!    Returns ordered set intersection of axes.
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        The ordered set intersection of axes
    //!-----------------------------------------------------------------------------------
    Axes operator&(const Axes&);

    //!-----------------------------------------------------------------------------------
    //! operator==
    //!    True if each ``Axis`` are matching and in same order (list comparison)
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        bool, True if each ``Axis`` are matching and in same order
    //!
    //!    See Also ``is_equal_set`` if you want the comparison to ignore the Axes order
    //!-----------------------------------------------------------------------------------
    bool operator==(const Axes&) const;

    //!-----------------------------------------------------------------------------------
    //! operator!=
    //!    The opposite of __eq__, True if not all ``Axis`` are matching or in
    //!    different order (list comparison)
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        bool, True if not all ``Axis`` are matching or in different order
    //!-----------------------------------------------------------------------------------
    bool operator!=(const Axes&) const;

    bool operator<(const Axes&) const;

    //!-----------------------------------------------------------------------------------
    //! axes
    //!    Axes considered nonzero if axes are nonzero.
    //!-----------------------------------------------------------------------------------
    //     def __nonzero__(self):
    //         """ """
    //         return bool(self._axes)

    // //!-----------------------------------------------------------------------------------
    // //! hash
    // //!-----------------------------------------------------------------------------------
    // size_t hash() const;

    //!-----------------------------------------------------------------------------------
    //! is_sub_set
    //!    Returns true if other is subset of self, i.e. <=
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        bool, true if other is subset of self
    //!-----------------------------------------------------------------------------------
    bool is_sub_set(const Axes& other) const;

    //!-----------------------------------------------------------------------------------
    //! is_super_set
    //!    Returns true if other is superset of self, i.e. >=
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        bool, true if other is superset of self
    //!-----------------------------------------------------------------------------------
    bool is_super_set(const Axes& other) const;

    //!-----------------------------------------------------------------------------------
    //! is_equal_set
    //!    Returns true if other has the same set of Axis names as self
    //!
    //!    Arguments:
    //!        other: the right-hand side operator axes
    //!
    //!    Returns:
    //!        bool, true if other has the same set of Axis names as self
    //!-----------------------------------------------------------------------------------
    bool is_equal_set(const Axes& other) const;

    //!-----------------------------------------------------------------------------------
    //! is_not_equal_set
    //!    Returns true if other does not the same set of Axis names as self
    //!
    //!    Arguments:
    //!       other: the right-hand side operator axes
    //!
    //!    Returns:
    //!       bool, true if other does not has the same set of Axis names as self
    //!-----------------------------------------------------------------------------------
    bool is_not_equal_set(const Axes& other) const;

    //!-----------------------------------------------------------------------------------
    //! T
    //!-----------------------------------------------------------------------------------
    //     def T(self):
    //         return Axes(axis.T for axis in self)

    //!-----------------------------------------------------------------------------------
    //! index
    //!    Returns the index of an axis
    //!
    //!    Arguments:
    //!        axis: The axis to search for.
    //!
    //!    Returns:
    //!        The index.
    //!-----------------------------------------------------------------------------------
    size_t index(const Axis&) const;

    //     @with_args_as_axes
    //!-----------------------------------------------------------------------------------
    //! assert_valid_broadcast
    //!    Checks whether axes can be broadcasted to new_axes. We require
    //!    that the components of axes be laid out in the same order in new_axes.
    //!
    //!    Axes:
    //!        axes: The original axes.
    //!        new_axes: The broadcasted axes.
    //!
    //!    Returns:
    //!        True if axes can be broadcasted to new_axes, False otherwise.
    //!-----------------------------------------------------------------------------------
    static void assert_valid_broadcast(const Axes& axes, const Axes& new_axes);

    //     @with_args_as_axes
    //!-----------------------------------------------------------------------------------
    //! is_valid_flatten_or_unflatten
    //!    Checks whether we can flatten OR unflatten from src_axes to dst_axes.
    //!
    //!    The requirements are that the components of axes should all be
    //!    present in new_axes and that they should be laid out in the same
    //!    order. This check is symmetric.
    //!-----------------------------------------------------------------------------------
    static bool is_valid_flatten_or_unflatten(const Axes& src_axes, const Axes& dst_axes);

    //     @with_args_as_axes
    //!-----------------------------------------------------------------------------------
    //! assert_valid_flatten
    //!    Checks whther axes can safely be flattened to produce new_axes.
    //!    The requirements are that the components of axes should all be
    //!    present in new_axes and that they should be laid out in the same
    //!    order.
    //!
    //!    Arguments:
    //!        unflattend_axes: The original axes.
    //!        flattened_axes: The flattened axes.
    //!
    //!    Returns:
    //!        True if axes can be safely flattened to new_axes, False otherwise.
    //!-----------------------------------------------------------------------------------
    static void assert_valid_flatten(const Axes& unflattend_axes, const Axes& flattened_axes);

    //     @with_args_as_axes
    //!-----------------------------------------------------------------------------------
    //! assert_valid_unflatten
    //!    Checks whether axes can safely be unflattened to produce new_axes.
    //!    The requirements are that the components of axes should all be
    //!    present in new_axes and that they should be laid out in the same
    //!    order.
    //!
    //!    Arguments:
    //!        flattened_axes: The original axes.
    //!        unflattend_axes: The unflattened axes.
    //!
    //!    Returns:
    //!        True if axes can be safely unflattened to new_axes, False otherwise.
    //!-----------------------------------------------------------------------------------
    static void assert_valid_unflatten(const Axes& flattened_axes, const Axes& unflattend_axes);

    //!-----------------------------------------------------------------------------------
    //! size
    //!    TODO: delete this method, the size should come from the tensor
    //!-----------------------------------------------------------------------------------
    size_t size() const;

    //!-----------------------------------------------------------------------------------
    //! operator<<
    //!-----------------------------------------------------------------------------------
    friend std::ostream& operator<<(std::ostream&, const Axes&);

    //!-----------------------------------------------------------------------------------
    //! as_nested_list
    //!    Converts Axes to a list of axes with flattened axes expressed as nested lists
    //!
    //!    Returns:
    //!        Nested list of Axis objects
    //!-----------------------------------------------------------------------------------
    static std::vector<Axis> as_nested_list(const Axes&);

    //!-----------------------------------------------------------------------------------
    //! as_flattened_list
    //!    Converts Axes to a list of axes with flattened axes expanded recursively.
    //!
    //!    Returns:
    //!        List of Axis objects
    //!-----------------------------------------------------------------------------------
    static std::vector<Axis> as_flattened_list(const Axes&);

    std::vector<Axis> convert(const Axes& ax);
    std::vector<Axis> convert(const std::vector<Axes>& ax);

private:
    void check_duplicates();
};

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
class AxesMap : public std::map<std::string, std::string>
{
public:
    AxesMap(const std::pair<std::string, std::string>&);
    AxesMap(std::initializer_list<std::pair<std::string, std::string>>);

    //--------------------------------------------------------------------------------------------
    // Returns:
    //     Axes with lengths from axes and names which have been passed through axes_map
    //--------------------------------------------------------------------------------------------
    Axes map_axes(const Axes&) const;

    //--------------------------------------------------------------------------------------------
    // Given a map from {old_axes_name: new_axes_name} and an old_axis map the
    // old_axis into the new_axes.
    //--------------------------------------------------------------------------------------------
    Axis map_axis(const Axis& old_axis) const;

private:
    std::map<std::string, std::set<std::string>> duplicate_axis_names();

    void assert_valid_axes_map();

public:
    //     def invert(self):
    //         return {v: k for k, v in self.items()}
};

//-----------------------------------------------------------------------------------------------
// _reduce_nested
//     Reduces a nested sequence by applying a function to each
//     of its elements and returns an aggregation.
//
//     Arguments:
//       elem: The object to be reduced, either a sequence
//         or a singleton.
//       agg: A variable holding information collected
//         as the sequence is collapsed.
//       func: A function to augment the aggregate by processing
//         a singleton. Should have the form func(agg, elem) -> agg
//
//     Returns:
//         agg: The final aggregate returned by the function.
//-----------------------------------------------------------------------------------------------
// def _reduce_nested(elem, agg, func):
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
class FlattenedAxis : public Axis
{
public:
    FlattenedAxis(const std::vector<Axis>& list, const std::string& new_name = "");

    virtual ~FlattenedAxis() {}

    //--------------------------------------------------------------------------------------------
    // Returns:
    //     True is this is a FlattendAxis.
    //--------------------------------------------------------------------------------------------
    bool is_flattened() const { return true; }

    //--------------------------------------------------------------------------------------------
    // Returns:
    //     Whether this axes contains no collapsed axes.
    //--------------------------------------------------------------------------------------------
    bool empty() const { return axes.size() == 0; }

    //--------------------------------------------------------------------------------------------
    // Returns:
    //     Whether this axes contains exactly one collapsed axes.
    //--------------------------------------------------------------------------------------------
    bool single() const { return axes.size() == 0; }

    bool operator==(const Axis& other) const;

    //     def __hash__(self):
    //         return hash(self.axes)

    friend std::ostream& operator<<(std::ostream&, const FlattenedAxis&);
    virtual std::string  to_string() const override;
    //     def __repr__(self):
    //         return 'FlattenedAxis(%s)' % ', '.join(repr(axis) for axis in self.axes)

    std::vector<Axis> axes;
};

//-----------------------------------------------------------------------------------------------
// default_dtype
//     Reduces a nested tuple describing the strides of a tensor
//     into a tuple giving the stride of each of its dimensions.
//
//     Arguments:
//         strides: The nested tuple.
//
//     Returns:
//         strides: The tuple of strides.
//-----------------------------------------------------------------------------------------------
// def reduce_strides(strides):
//     return tuple(int(_reduce_nested(elem, float('inf'), min))
//                  for elem in strides)

//-----------------------------------------------------------------------------------------------
// _make_stride
//     Generates a nested tuple that provides the striding information
//     for an occurrence of axis. If the axis is a FlattenedAxis, the
//     stride will be a tuple containing the strides of each collapsed
//     axis. Otherwise, the stride will be an integer.
//
//     Arguments:
//         inner_size: The total size of all dimensions smaller than this
//         axis, i.e. all axes to the right of this one when they are
//         laid out in c-contiguous order.
//         axis: The axis for which we are generating a stride.
//         fsz: A nested tuple supplying the sizes of each dimension collapsed
//         into the axis. The size may be larger than the length of the axis.
//
//     Returns:
//         inner_size: The total size of this axis and all smaller dimensions.
//         stride: The stride given to the axis.
//-----------------------------------------------------------------------------------------------
// def _make_stride(inner_size, axis, fsz):
//     if axis.is_flattened:
//         return _make_strides(inner_size, axis.axes, fsz)
//     else:
//         stride = inner_size
//         inner_size *= fsz
//         return inner_size, stride

//-----------------------------------------------------------------------------------------------
// _make_strides
//     Generates a tuple of strides for a set of axes. See _make_stride
//     for a description of the stride given to each axis.
//
//     Arguments:
//         inner_size: The total size of all dimensions smaller than
//         the axes.
//         axes: The axes for which we are generating strides.
//         full_sizes: The size of each axis.
//
//     Returns:
//         inner_size: The total size of these axes and all smaller dimensions.
//         strides: The strides generated for the axes.
//-----------------------------------------------------------------------------------------------
// def _make_strides(inner_size, axes, full_sizes):
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
class TensorDescription : public NameableValue
{
public:
    //!-----------------------------------------------------------------------------------
    //! constructor
    //!-----------------------------------------------------------------------------------
    TensorDescription(op_ptr                 op    = nullptr,
                      const Axes&            _axes = Axes(),
                      tensor_description_ptr base  = nullptr,
                      //   layout,
                      ElementType           et                     = element_type_float,
                      ngraph::tensor_stride full_strides           = ngraph::tensor_stride(),
                      ngraph::tensor_size   full_sizes             = ngraph::tensor_size(),
                      size_t                offset                 = 0,
                      TensorDescription*    next_tensor_decription = nullptr,
                      const std::string&    name                   = "",
                      bool                  is_persistent          = false,
                      bool                  is_input               = false,
                      bool                  is_placeholder         = false);

    // std::string name() const;
    std::vector<size_t>    shape() const;
    tensor_description_ptr base() const;
    ElementType            element_type() const;
    size_t                 tensor_size() const;

    //!-----------------------------------------------------------------------------------
    //! operator<<
    //!-----------------------------------------------------------------------------------
    //     def __repr__(self):
    //         return self.base.name

    //!-----------------------------------------------------------------------------------
    //! is_persistent
    //!    Returns: True if persists from computation to computation.
    //!-----------------------------------------------------------------------------------
    bool is_persistent() const;

    //!-----------------------------------------------------------------------------------
    //! is_input
    //!    Returns: True if writable from host.
    //!-----------------------------------------------------------------------------------
    bool is_input() const;

    //!-----------------------------------------------------------------------------------
    //! is_placeholder
    //!    Returns: True if a placeholder; a place to attach a tensor.
    //!-----------------------------------------------------------------------------------
    bool is_placeholder() const;

    //!-----------------------------------------------------------------------------------
    //! parameter_key
    //!    Returns: A tuple that can be used to tell if two views of a tensor are equivalent.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def parameter_key(self):
    //         return (self.shape, self.dtype, self.offset, self.strides, self.layout)
    size_t parameter_key() const;

    //!-----------------------------------------------------------------------------------
    //! axes_key
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def axes_key(self):
    //         return (self.axes, self.shape, self.dtype, self.offset, self.strides, self.layout)
    axes_key_t axes_key() const;

    //!-----------------------------------------------------------------------------------
    //! flatten
    //!    Flattens a tensor description to give it the Axes in new_axes.
    //!    See Axes.assert_valid_flatten for a description of permitted values of new_axes.
    //!
    //!    Arguments:
    //!        new_axes: The Axes of the flattened tensor description.
    //!
    //!    Returns:
    //!        The reshaped tensor description.
    //!-----------------------------------------------------------------------------------
    //     def flatten(self, new_axes):
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

    //!-----------------------------------------------------------------------------------
    //! unflatten
    //!    Unflattens a tensor description to give it the Axes in new_axes.
    //!    See Axes.assert_valid_unflatten for a description of the permitted values of
    //!    new_axes
    //!
    //!    Arguments:
    //!        new_axes: The Axes of the unflattened TensorDescription.
    //!
    //!    Returns:
    //!        The unflattened tensor description.
    //!-----------------------------------------------------------------------------------
    //     def unflatten(self, new_axes):
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

    //!-----------------------------------------------------------------------------------
    //! transpose
    //!    Reverses the axes of the tensor description.
    //!
    //!    Retuns:
    //!        A tensor description with the axes reversed.
    //!-----------------------------------------------------------------------------------
    //     def transpose(self):
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

    //!-----------------------------------------------------------------------------------
    //! clone
    //!    Creates a copy of this tensor description
    //!
    //!    Retuns:
    //!        A copy of this tensor description
    //!-----------------------------------------------------------------------------------
    //     def clone(self):
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

    //!-----------------------------------------------------------------------------------
    //! broadcast
    //!    Adds axes to a tensor description to give it a new shape.
    //!    See Axes.assert_valid_broadcast for a description of the permitted
    //!    transformations.
    //!
    //!    Arguments:
    //!        new_axes: The axes of the broadcasted tensor description.
    //!
    //!    Returns:
    //!        TensorDescription: The broadcasted tensor description.
    //!-----------------------------------------------------------------------------------
    TensorDescription broadcast(const Axes& new_axes);

    //!-----------------------------------------------------------------------------------
    //! reorder
    //!    Shuffles axes of a tensor to give it a new shape. The axes of
    //!    this tensor description and new_axes must have the same elements.
    //!
    //!    Arguments:
    //!        new_axes: The axes of the reordered tensor.
    //!
    //!    Returns:
    //!        TensorDescription: The reordered tensor description.
    //!-----------------------------------------------------------------------------------
    //     def reorder(self, new_axes):
    //         if not self.axes.is_equal_set(new_axes):
    //             raise ValueError((
    //                 "Reorder can't change which axes are available, only the "
    //                 "order.  {} and {} are different sets, not just order."
    //             ).format(self, new_axes))

    //         return self.reorder_and_broadcast(new_axes)

    //!-----------------------------------------------------------------------------------
    //! reorder_and_broadcast
    //!    Adds or shuffles axes to give a tensor description a new shape.
    //!    This function is used to implement broadcast and reorder.
    //!
    //!    Arguments:
    //!        new_axes: The axes of the broadcasted or reordered tensor.
    //!
    //!    Returns:
    //!        TensorDescription: A description of the tensor after the
    //!        transformation.
    //!-----------------------------------------------------------------------------------
    TensorDescription reorder_and_broadcast(const Axes& new_axes);
    //     def reorder_and_broadcast(self, new_axes):
    //         def zero_in_shape(tup):
    //             if isinstance(tup, collections.Iterable):
    //                 return tuple(
    //                     zero_in_shape(t) for t in tup
    //                 )
    //             else:
    //                 return 0

    //         new_axes = Axes(new_axes)
    //         new_strides = []
    //         new_sizes = []
    //         for axis in new_axes:
    //             if axis in self.axes:
    //                 idx = self.axes.index(axis)
    //                 new_strides.append(self.full_strides[idx])
    //                 new_sizes.append(self.full_sizes[idx])
    //             elif axis.is_flattened:
    //                 lengths = axis.axes.full_lengths
    //                 new_strides.append(zero_in_shape(lengths))
    //                 new_sizes.append(lengths)
    //             else:
    //                 new_strides.append(0)
    //                 new_sizes.append(axis.length)

    //         return TensorDescription(
    //             new_axes,
    //             base=self.base,
    //             dtype=self.dtype,
    //             full_strides=new_strides,
    //             full_sizes=new_sizes,
    //             offset=self.offset,
    //             next_tensor_description=self,
    //             name=self.name + 'rReorderBroadcast',
    //         )

    //!-----------------------------------------------------------------------------------
    //! cast
    //!    Return a tensor desciption for a view of the tensor.
    //!
    //!    Arguments:
    //!        new_axes: The axes for the view.
    //!
    //!    Returns:
    //!        The tensor description.
    //!-----------------------------------------------------------------------------------
    //     def cast(self, new_axes):
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

    //!-----------------------------------------------------------------------------------
    //! slice
    //!    Return a tensor description for a slice view of this tensor.
    //!
    //!    Arguments:
    //!        slices: The slices to take from the tensor, each of which is
    //!        either an integer or a python slice. If the input has too few
    //!        axes for the tensor, we assume that the entire axis should be
    //!        taken for dimensions towards the end of the tensor.
    //!        new_axes: the axes to use as labels for the sliced tensor.
    //!
    //!    Returns:
    //!        The tensor description for the slice.
    //!-----------------------------------------------------------------------------------
    //     def slice(self, slices, new_axes):
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

    //!-----------------------------------------------------------------------------------
    //! shape
    //!    Returns: The shape of the tensor.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def shape(self):
    //         return self.axes.lengths

    //!-----------------------------------------------------------------------------------
    //! strides
    //!    The strides of the tensor.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def strides(self):
    //         return reduce_strides(self.full_strides)

    //!-----------------------------------------------------------------------------------
    //! sizes
    //!    The allocated sizes for each axis.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def sizes(self):
    //         return tuple(_reduce_nested(_, 1, operator.mul)
    //                      for _ in self.full_sizes)

    //!-----------------------------------------------------------------------------------
    //! tensor_size
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def tensor_size(self):
    //         result = self.dtype.itemsize
    //         for s in self.sizes:
    //             result = result * s
    //         return result

    //!-----------------------------------------------------------------------------------
    //! c_contiguous
    //!    Returns:
    //!        True if the tensor's strides are row-major contiguous.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def c_contiguous(self):
    //         s = self.dtype.itemsize
    //         cstrides = []
    //         for _ in reversed(self.shape):
    //             cstrides.insert(0, s)
    //             s = s * _
    //         return tuple(cstrides) == self.strides

    //!-----------------------------------------------------------------------------------
    //! broadcast_contiguous
    //!    Returns:
    //!        True if tensor's strides are contiguous or broadcasted
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def broadcast_contiguous(self):
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

    //!-----------------------------------------------------------------------------------
    //! base
    //!    The viewed tensor description or None if not a view.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def base(self):
    //         return self.__base or self

    //!-----------------------------------------------------------------------------------
    //! layout
    //!    The layout of the underlying storage.
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def layout(self):
    //         return self.__layout

    //!-----------------------------------------------------------------------------------
    //! layout
    //!    Sets the backend-specific memory layout to be used by the tensor.
    //!
    //!    Arguments:
    //!      value: the layout to use
    //!
    //!    Returns:
    //!-----------------------------------------------------------------------------------
    //     @layout.setter
    //     def layout(self, value):
    //         self.__layout = value

    //!-----------------------------------------------------------------------------------
    //! register
    //!-----------------------------------------------------------------------------------
    //     @property
    //     def register(self):
    //         return self.base.__register

    //!-----------------------------------------------------------------------------------
    //! register
    //!-----------------------------------------------------------------------------------
    //     @register.setter
    //     def register(self, value):
    //         self.base.__register = value

    //!-----------------------------------------------------------------------------------
    //! is_base
    //!    This tensor provides its own storage.
    //!-----------------------------------------------------------------------------------
    //     def is_base(self):
    //         return self.__base is None

    op_ptr                 op;
    Axes                   axes;
    bool                   __is_persistent;
    bool                   __is_input;
    bool                   __is_placeholder;
    tensor_description_ptr __base;

    // __layout = layout
    // __value = None
    // __buffer = None
    // __register = None
    ElementType            dtype;
    size_t                 offset;
    size_t                 ndim;
    ngraph::tensor_size    full_sizes;
    ngraph::tensor_stride  full_strides;
    tensor_description_ptr next_tensor_description;
};

} // end of namespace ngraph

namespace std
{
    template <>
    struct std::hash<ngraph::Axis>
    {
        size_t operator()(const ngraph::Axis& axis) const
        {
            std::hash<std::string> h1;
            std::hash<size_t>      h2;
            return ngraph::hash_combine({h1(axis.name), h2(axis.length())});
        }
    };
}

namespace std
{
    template <>
    struct std::hash<ngraph::Axes>
    {
        size_t operator()(const ngraph::Axes& axes) const
        {
            std::hash<ngraph::Axis>     h1;
            std::vector<size_t> hashes;
            for (auto axis : axes)
            {
                hashes.push_back(h1(axis));
            }
            return ngraph::hash_combine(hashes);
        }
    };
}
