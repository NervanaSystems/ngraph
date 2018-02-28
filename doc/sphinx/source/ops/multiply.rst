.. multiply.rst:

########
Multiply
########

.. code-block:: cpp

   Multiply  //  Elementwise multiply operation


Description
===========

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is the product of the
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

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \texttt{arg0}_{i_0, \ldots, i_{n-1}} \texttt{arg1}_{i_0, \ldots, i_{n-1}}

Backprop
========

.. math::

   \overline{\texttt{arg0}} &\leftarrow \Delta\ \texttt{arg1}\\
   \overline{\texttt{arg1}} &\leftarrow \Delta\ \texttt{arg0}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Multiply
   :project: ngraph
   :members:
