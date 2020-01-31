.. sigmoid.rst:

#######
Sigmoid
#######

.. code-block:: cpp

   Sigmoid  // Elementwise sigmoid operation


Description
===========

Produces a single output tensor of the same element type and shape as
``arg,`` where the value at each coordinate of ``output`` is the
sigmoid  of ``arg`` at the same coordinate.


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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \frac{1}{1+\exp(-\mathtt{arg}_{i_0,
   \ldots, i_{n-1}})}


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Sigmoid
   :project: ngraph
   :members:
