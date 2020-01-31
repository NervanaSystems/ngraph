.. tan.rst:

###
Tan
###

.. code-block:: cpp

   Tan  //  Elementwise tangent operation


Description
===========

Produces a tensor of the same element type and shape as ``arg,``
where the value at each coordinate of ``output`` is the tangent
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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \tan(\mathtt{arg}_{i_0, \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \frac{\Delta}{\cos^2(\mathtt{arg})}

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Tan
   :project: ngraph
   :members:
