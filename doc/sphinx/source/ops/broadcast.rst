.. broadcast.rst:

#########
Broadcast
#########

Description
===========

Operation whose ``output`` tensor that ignores axes not in the ``arg``
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

+---------------------+---------------+
| Name                | Type          |
+=====================+===============+
| ``shape``           | ``Shape``     |
+---------------------+---------------+
| ``broadcast_axes``  | ``AxisSet``   |
+---------------------+---------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``shape``.             |
+-----------------+-------------------------+--------------------------------+

The shape of ``arg`` must match ``shape`` with elements in ``broadcast_axes`` removed.


Mathematical Definition
=======================

For a coordinate :math:`C`, let :math:`p(C)` be a coordinate with the
axes in ``broadcast_axes`` removed.  For example, if
:math:`\texttt{broadcast_axes}=\{1,3\}` then :math:`p([d_0, d_1,
d_2, d_3, d_4]) = [d_0, d_2, d_4]`.  Then

.. math::

   \texttt{output}_C = \texttt{arg}_{p(C)}.
   


Backprop
========

.. math::

   \overline{\texttt{arg}} \leftarrow \texttt{Sum}(\Delta, \texttt{broadcast_axes}).
   

C++ Interface
=============

.. doxygenclass:: ngraph::op::Broadcast
   :members:
