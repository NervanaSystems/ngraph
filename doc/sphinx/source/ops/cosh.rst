.. cosh.rst:

####
Cosh
####

.. code-block:: cpp

   Cosh  //  Elementwise hyperbolic cosine operation


Description
===========

Produces a tensor of the same element type and shape as ``arg``, where
the value at each coordinate of ``output`` is the hyperbolic cosine of
the value at the corresponding coordinate of ``arg``.

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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \cosh(\mathtt{arg}_{i_0, \ldots, i_{n-1}})


Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \Delta\ \sinh(\mathtt{arg})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Cosh
   :project: ngraph
   :members:
