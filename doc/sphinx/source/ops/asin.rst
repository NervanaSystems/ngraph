.. asin.rst:

####
Asin
####

.. code-block:: cpp

   Asin  // Elementwise asin operation


Description
===========

Produces a tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the inverse sine of the
value at the corresponding coordinate of ``arg``.

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
| ``output``      | Same as ``arg``         | Same as ``arg``.               |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \sin^{-1}(\mathtt{arg}_{i_0, \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \frac{\Delta}{\sqrt{1-\mathtt{arg}^2}}

C++ Interface
=============

.. doxygenclass:: ngraph::op::Asin
   :project: ngraph
   :members:
