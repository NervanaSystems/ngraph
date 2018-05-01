.. power.rst:

#####
Power
#####

.. code-block:: cpp

   Power  // Elementwise exponentiation operation


Description
===========

Elementwise exponentiation operation.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg0``        | any                     | any                            |
+-----------------+-------------------------+--------------------------------+
| ``arg1``        | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{arg0}_{i_0, \ldots, i_{n-1}} ^ {\mathtt{arg1}_{i_0, \ldots, i_{n-1}}}

Backprop
========

.. math::

   \overline{\mathtt{arg0}} &\leftarrow \frac{\Delta \cdot \mathtt{arg1}}{\mathtt{arg0}} \\
   \overline{\mathtt{arg1}} &\leftarrow \Delta \cdot \mathtt{output} \cdot \log(\mathtt{arg1})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Power
   :project: ngraph
   :members:
