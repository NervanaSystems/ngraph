.. cos.rst:

###
Cos
###

.. code-block:: cpp

   Cos  //  Elementwise cosine operation


Description
===========

Produces a tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the cosine of the
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
| ``output``      | Same as ``arg``         | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \cos(\mathtt{arg}_{i_0, \ldots, i_{n-1}})


Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow -\Delta\ \sin(\mathtt{arg})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Cos
   :project: ngraph
   :members:
