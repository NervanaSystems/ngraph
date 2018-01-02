.. axes.rst: 

Axes
####

Introduction
============

An ``Axis`` (``ngraph.op_graph.axes.Axis``) labels a dimension of a tensor.
The op-graph uses the identity of ``Axis`` objects to pair and specify
dimensions in symbolic expressions. This system has several advantages over
using the length and position of the axis as in other frameworks:

1. **Convenience.** The dimensions of tensors, which may be nested
deep in a computation graph, can be specified without having to
calculate their lengths.

2. **Safety.** Axis labels are analogous to types in general-purpose
programming languages, allowing objects to interact only when
they are permitted to do so in advance. In symbolic computation,
this prevents interference between axes that happen to have the
same lengths but are logically distinct. For example, if the number of
training examples and the number of input features are both 50.

3. **Generic.** The order of axes for multidimensional tensors does not
imply a specific data layout or striding, making the graph specification
compatible across different hardware with different constraints.

Core concepts
=============

Axis and axes
-------------

The ``Axis`` object represents one dimension of a tensor, and can be created 
with the ``ng.make_axis`` method.

  ::

    H = ng.make_axis(length=3, name='height')
    W = ng.make_axis(length=4, name='width')

For tensors with multiple dimensions, we create an ``Axes`` passing in a list 
of individual ``Axis`` objects. 

.. Note::
   The ordering does *not* matter in specifying the axes, and has no bearing on 
   the eventual data layout during execution. Refer to the Properties section 
   below for a full description of axes properties.

::

  axes = ng.make_axes([H, W])

We use ``Axes`` to define the shape of tensors in Intel nGraph library. For 
example:

  ::

    image = ng.placeholder(axes)

We can also delay the specification of the axis length.

  ::

    H = ng.make_axis(name='height')
    W = ng.make_axis(name='width')
    image = ng.placeholder([H, W])
    H.length = 3
    W.length = 4

Semantics
=========

In Intel Nervana graph, our axis design is very flexible. Axes can be given 
arbitrary names and the ordering of the axes does not matter. Sometimes, 
however, axes need to have additional semantic information provided to 
operations.

Properties
==========

1. The order of axes does not matter.

  - Two tensors ``x`` and ``y`` are considered having the same type if:
    - ``x`` and ``y`` have the same number of axes and same set of axes.
    - After shuffling of ``y``'s axes to be the same order of ``x``'s, the 
    underlying values are the same.

  - We can check element-wise tensor equality using ``ng.equal()``. ::

      import numpy as np
      import ngraph as ng
      import ngraph.transformers as ngt

      H = ng.make_axis(length=2)
      W = ng.make_axis(length=3)
      np_val = np.random.rand(2, 3)
      x = ng.constant(np_val, [H, W])
      y = ng.constant(np_val.T, [W, H])
      z = ng.equal(x, y)

      trans = ngt.make_transformer()
      comp = trans.computation([z])
      z_val = comp()[0]
      print(z_val)
      # [[ True  True  True]
      #  [ True  True  True]]

2. An axis can occur at most once in the axes of a tensor.

  For example: ::

      H = ng.make_axis(length=2)
      W = ng.make_axis(length=2)
      x = ng.constant(np.ones((2, 2)), [H, H])  # throws exception
      x = ng.constant(np.ones((2, 2)), [H, W])  # good

3. Axes have context. A set of standard neon™ axes are defined for neon frontends.

  - Axes roles:

  ::

    ar = Namespace()
    ar.Height = ng.make_axis_role()
    ar.Width = ng.make_axis_role()
    ar.Depth = ng.make_axis_role()
    ar.Channel = ng.make_axis_role()
    ar.Channelout = ng.make_axis_role()
    ar.Time = ng.make_axis_role()

  - Image / feature map:

  ::

    ax = Namespace()
    ax.N = ng.make_axis(name='N', docstring="minibatch size")
    ax.C = ng.make_axis(roles=[ar.Channel], docstring="number of input feature maps")
    ax.D = ng.make_axis(roles=[ar.Depth], docstring="input image depth")
    ax.H = ng.make_axis(roles=[ar.Height], docstring="input image height")
    ax.W = ng.make_axis(roles=[ar.Width], docstring="input image width")

  - Filter (convolution kernel):

  ::

    ax.R = ng.make_axis(roles=[ar.Height], docstring="filter height")
    ax.S = ng.make_axis(roles=[ar.Width], docstring="filter width")
    ax.T = ng.make_axis(roles=[ar.Depth], docstring="filter depth")
    ax.J = ng.make_axis(roles=[ar.Channel], docstring="filter channel size (for crossmap pooling)")
    ax.K = ng.make_axis(roles=[ar.Channelout], docstring="number of output feature maps")

  - Output:

  ::

    ax.M = ng.make_axis(roles=[ar.Depth], docstring="output image depth")
    ax.P = ng.make_axis(roles=[ar.Height], docstring="output image height")
    ax.Q = ng.make_axis(roles=[ar.Width], docstring="output image width")

  - Recurrent:

  ::

    ax.REC = ng.make_axis(name='R', roles=[ar.Time], docstring="recurrent axis")

  - Target:

  ::

    ax.Y = ng.make_axis(docstring="target")


Axes operations
===============

``Axes`` (``ngraph.op_graph.axes.Axes``) has ``list`` and ``set`` behaviors at 
the same time. ``Axes`` are internally stored and can be used as ``list``, while 
we also have use cases of ``Axes`` as ``set``. Here's a list of supported 
operations by ``Axes`` and their expected behavors.

- ``__add__``: list operation, concatenated axes, throws exception when there
  are axis duplications
- ``__sub__``: set operation, returns the ordered set difference of axes
- ``__or__``: set operation, returns ordered set union of axes
- ``__and__``: set operation, returns ordered set intersection of axes
- ``__eq__``: list operation, true if each ``Axis`` are matching and in same
  order
- ``__ne__``: list operation, true if not all ``Axis`` are matching or in
  different order
- ``is_sub_set``, ``is_super_set``, ``is_equal_set`` and ``is_not_equal_set``:
  set operations


Element-wise binary ops
=======================

- When axes match, output the same axes. 

::

  (H,) + (H,) -> (H,)
  (H, W) + (H, W) -> (H, W)

- Automatic broadcasting / dim shuffle, the output axis order is determined by 
  the input axis order of the left and right operands. 

  ::

  (H, W) + (H,) -> (H, W)
  (H, W) + (W,) -> (H, W)
  (H, W) + (W, N) -> (H, W, N)
  (H, W) + (N, W) -> (H, W, N)
  (C, H) + (W, H, N) -> (C, H, W, N)

  Axis order is determined by the following rules:

  1. If the set of axes for both operands match exactly, but the order is 
     different, use the order of the left operand.
  2. If one operand's axes are a superset of the other's, use that operand's 
     axis order
  3. Otherwise the order is determined by concatenating the left operand's axes 
     with the axes from the right operand that are not present in the left 
     operand (left_axes + (right_axes - left_axes)). 

  ::

  (H, W, N) + (N, H) -> (H, W, N)
  (H, W) + (N, H, W) -> (N, H, W)
  (H, W) + (N, W, H) -> (N, W, H)
  (C, H, W) + (N, W, H) -> (C, H, W, N)
  (N, C, H, W) + (C, H, W, N) -> (N, C, H, W)

- Commutative property is as usual, although the axis order of the equivalent 
  tensors can be different. 

::

  (H,) + (W,) -> (H, W)
  (W,) + (H,) -> (W, H)
  (C,) + (H, W) -> (C, H, W)
  (H, W) + (C,) -> (H, W, C)

  In the following example, ``z`` from left and right are equivalent, although 
  the axis orders are different.

  ::

    x = ng.constant(np.ones((2, 3)), [H, W]) | x = ng.constant(np.ones((2, 3)), [H, W])
    y = ng.constant(np.ones((3, 2)), [W, H]) | y = ng.constant(np.ones((3, 2)), [W, H])
    z = x + y                                | z = y + x  # <== changed order
                                             |
    trans = ngt.make_transformer()           | trans = ngt.make_transformer()
    comp = trans.computation([z])            | comp = trans.computation([z])
    z_val = comp()[0]                        | z_val = comp()[0]
    print(z_val)                             | print(z_val)
    print(z_val.shape)                       | print(z_val.shape)
    -----------------------------------------------------------------------------
    Output:                                  | Output:
    [[ 2.  2.  2.]                           | [[ 2.  2.]
     [ 2.  2.  2.]]                          |  [ 2.  2.]
    (2, 3)                                   |  [ 2.  2.]]
                                             | (3, 2)

- Associative property is as usual. 

::

  ((H,) + (W,)) + (N,) -> (H, W) + (N,) -> (H, W, N)
  (H,) + ((W,) + (N,)) -> (H,) + (W, N) -> (H, W, N)

- Distributive property is as usual. 

::

  (H,) * ((W,) + (N,)) = (H,) * (W, N) = (H, W, N)
  (H,) * (W,) + (H,) * (N,) = (H, W) * (H, N) = (H, W, N)


Dot operation
-------------

When two tensors are provided to a multiaxis operation, such as ``ng.dot()``,
we need to indicate the corresponding axes that should be paired together.

For example:

  ::

    # 2d dot
    (H, W) • (W, N) -> (H, N)

    # 4d dot
    (M, C, H, W) • (C, H, W, N) -> (M, N)

    # swapping the order of the axes is allowed
    (M, C, H, W) • (C, H, W, N) -> (M, N)
    (M, W, H, C) • (C, H, W, N) -> (M, N)


Axes reduction
==============

- We specify the reduction axes in ``reduction_axes``. Reduction operations can
  have an arbitrary number of reduction axes. The order of the reduction axes
  can be arbitrary.
- When ``reduction_axes`` is empty, reduction is performed on *none* of the axes.

Examples: ::

    ax_C = ng.make_axis(name="C", docstring="number of input feature maps")
    ax_H = ng.make_axis(name="H", docstring="input image height")
    ax_W = ng.make_axis(name="W", docstring="input image width")
    x = ng.placeholder([ax_C, ax_H, ax_W])
    ng.sum(x, reduction_axes=[])            #-> [C, H, W]
    ng.sum(x, reduction_axes=[ax_C])        #-> [H, W]
    ng.sum(x, reduction_axes=[ax_C, ax_W])  #-> [H]
    ng.sum(x, reduction_axes=[ax_W, ax_C])  #-> [H]
    ng.sum(x, reduction_axes=x.axes)        #-> []


Axes casting
============

Use ``ng.cast_axes`` to cast at axes to targeting axes with the same dimensions.
For example, we might want to sum two layers' outputs, where they have the same
dimensions but different axes. Examples are shown below: ::

    # assume C1.length == C2.length == 100
    hidden_1 = ng.constant(np.ones((100, 128)), [C1, N])
    hidden_2 = ng.constant(np.ones((100, 128)), [C2, N])

    # if we add directly without casting
    sum_direct = hidden_1 + hidden_2  # sum_direct has axes: [C1, C2, N]

    # cast before sum
    hidden_2_cast = ng.cast_axes(hidden_2, [C1, N])
    sum_cast = hidden_1 + hidden_2_cast  # sum_cast has axes: [C1, N]


Axes broadcasting
=================

Use ``ng.broadcast`` to broadcast to new axes. The new axes must be a superset
of the original axes. The order of the new axes can be arbitrary. For example: ::

    x = ng.placeholder([ax_C, ax_H])
    ng.broadcast(x, [ax_C, ax_H, ax_W])  #-> [C, H, W]
    ng.broadcast(x, [ax_W, ax_H, ax_C])  #-> [W, H, C]


Axes reordering
===============

Use ``ng.axes_with_order`` to reorder axes. The new axes must be the same set as the
original axes. Examples:


.. code-block:: python

    from ngraph.frontends.neon.axis import ax
     x = ng.placeholder([ax.C, ax.H, ax.W])
     ng.axes_with_order(x, [ax.H, ax.W, ax.C])

