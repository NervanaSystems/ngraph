.. relu.rst:

####
Relu
####

.. code-block:: cpp

   Relu  // Elementwise relu operation

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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} =
   \begin{cases}
   0&\text{if }\mathtt{arg}_{i_0, \ldots, i_{n-1}} \le 0 \\
   \mathtt{arg}_{i_0, \ldots, i_{n-1}}&\text{otherwise}
   \end{cases}

C++ Interface
=============

.. doxygenclass:: ngraph::op::Relu
   :project: ngraph
   :members:
