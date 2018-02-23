.. minimum.rst:

#######
Minimum
#######

.. code-block:: cpp

   Minimum  // Short description.


Description
===========

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is the minimum of the
values at the corresponding input coordinates.

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

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \min(\texttt{arg0}_{i_0, \ldots, i_{n-1}}, \texttt{arg1}_{i_0, \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{\texttt{arg0}} &\leftarrow \texttt{Less}(\texttt{arg0}, \texttt{arg1})\ \Delta \\
   \overline{\texttt{arg1}} &\leftarrow \texttt{Less}(\texttt{arg1}, \texttt{arg0})\ \Delta


C++ Interface
=============

.. doxygenclass:: ngraph::op::Minimum
   :project: ngraph
   :members:
