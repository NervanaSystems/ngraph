.. sqrt.rst:

####
Sqrt
####

.. code-block:: cpp

   Sqrt  //  Elementwise square root operation


Description
===========

Produces a tensor of the same element type and shape as ``arg,``
where the value at each coordinate of ``output`` is the square root
of the value at the corresponding coordinate of ``arg.``

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \sqrt{\mathtt{arg}_{i_0, \ldots, i_{n-1}}}

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \frac{\Delta}{2\cdot \mathtt{output}}

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Sqrt
   :project: ngraph
   :members:
