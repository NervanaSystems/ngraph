.. exp.rst:

###
Exp
###

.. code-block:: cpp

   Exp  // Elementwise expine operation


Description
===========

Produces a tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the expine of the
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

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \exp(\texttt{arg}_{i_0, \ldots, i_{n-1}})


Backprop
========

.. math::

   \overline{\texttt{arg}} \leftarrow \Delta\ \texttt{output}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Exp
   :project: ngraph
   :members:
