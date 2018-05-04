.. broadcast.rst:

#########
Broadcast
#########

.. code-block:: cpp

   Broadcast  // Operation that produces a tensor based on arg's axes


Description
===========

Operation whose ``output`` tensor ignores axes not in the ``arg``
tensor.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Attributes
----------

+---------------------+---------------+------------------------------------+
| Name                | Type          | Notes                              |
+=====================+===============+====================================+
| ``shape``           | ``Shape``     | The shape of the output.           |
+---------------------+---------------+------------------------------------+
| ``broadcast_axes``  | ``AxisSet``   | Axis positions in ``shape`` that   |
|                     |               | are broadcast.                     |
+---------------------+---------------+------------------------------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``shape``              |
+-----------------+-------------------------+--------------------------------+

The shape of ``arg`` must match ``shape`` with elements in ``broadcast_axes`` removed.


For example, if ``arg`` is :math:`[a, b, c]` then

.. math::

   \mathtt{Broadcast(arg, Shape{2, 3}, AxisSet{0})} &=
   \begin{bmatrix}
   a & b & c\\
   a & b & c
   \end{bmatrix}\\
   \mathtt{Broadcast(arg, Shape{3, 2}, AxisSet{1})} &=
   \begin{bmatrix}
   a & a\\
   b & b\\
   c & c
   \end{bmatrix}


Mathematical Definition
=======================

For a coordinate :math:`C`, let :math:`p(C)` be a coordinate with the
axes in ``broadcast_axes`` removed.  For example, if
:math:`\mathtt{broadcast_axes}=\{1,3\}` then :math:`p([d_0, d_1,
d_2, d_3, d_4]) = [d_0, d_2, d_4]`.  Then

.. math::

   \mathtt{output}_C = \mathtt{arg}_{p(C)}.
   


Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \mathtt{Sum}(\Delta, \mathtt{broadcast_axes}).
   

C++ Interface
=============

.. doxygenclass:: ngraph::op::Broadcast
   :project: ngraph
   :members:
