.. sign.rst:

####
Sign
####

.. code-block:: cpp

   Sign  //  Elementwise sign operation


Description
===========

Produces a tensor of the same element type and shape as ``arg,``
where the value at each coordinate of ``output`` is the sign (-1, 0, 1)
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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{sgn}(\mathtt{arg}_{i_0, \ldots, i_{n-1}})

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Sign
   :project: ngraph
   :members:
