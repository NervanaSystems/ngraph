.. concat.rst:

######
Concat
######

.. code-block:: cpp

   Concat  // Concatenation operation 


Description
===========

Produces a single output tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the absoloute value of the
value at each ``arg`` coordinate.

Inputs
------

+-----------------+-----------------+------------------------------------------------------+
| Name            | Type            | Notes                                                |
+=================+=================+======================================================+
| ``args``        | ``Nodes``       | All element types the same.                          |
|                 |                 | All shapes the same except on ``concatenation_axis`` |
+-----------------+-----------------+------------------------------------------------------+

Attributes
----------

+-------------------------+----------------------------------+
| Name                    | Notes                            |
+=========================+==================================+
| ``concatenation_axis``  | Less than the rank of the shape  |
+-------------------------+----------------------------------+

Outputs
-------

+-----------------+-------------------------+----------------------------------------------------+
| Name            | Element Type            | Shape                                              |
+=================+=========================+====================================================+
| ``output``      | Same as ``args``         | Same as ``arg`` on non-``concatenation_axis``     |
|                 |                          | Sum of ``concatenation_axis`` lengths of ``args`` |
+-----------------+-------------------------+----------------------------------------------------+


Mathematical Definition
=======================

We map each tensor in ``args`` to a segment of ``output`` based on the
coordinate at ``coordinate_axis``.

Let

.. math::

   s(i) &= \sum_{j<i} \mathtt{args}[i].\mathtt{shape}\left[\mathtt{concatenation_axis}\right]\\
   t(i) &= \text{The greatest }j\text{ such that }i \ge s(j)\\
   p(C)_i &= \begin{cases}
   C_i-s(t(i))&\text{if }i==\mathtt{concatenation_axis}\\
   C_i&\text{otherwise}
   \end{cases}\\
   \mathtt{output}_C&=\mathtt{args}[t(C_i)]_{p(C)}



Backprop
========

We slice the backprop value into the backprops associated with the inputs.


C++ Interface
=============

.. doxygenclass:: ngraph::op::Concat
   :project: ngraph
   :members:
