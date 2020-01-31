.. one_hot.rst:

######
OneHot
######

.. code-block:: cpp

   OneHot  // One-hot expansion


Description
===========

Inputs
------

+-----------------+-------------------+---------------------------------------------------------+
| Name            | Element Type      | Shape                                                   |
+=================+===================+=========================================================+
| ``arg``         | Any integral type | :math:`d_1,\dots,d_{m-1},d_{m+1},\dots,d_n)~(n \geq 0)` |
+-----------------+-------------------+---------------------------------------------------------+

Attributes
----------

+------------------+----------------------------------------------------------------+
| Name             | Description                                                    |
+==================+================================================================+
| ``shape``        | The desired output shape, including the new one-hot axis.      |
+------------------+----------------------------------------------------------------+
| ``one_hot_axis`` | The index within the output shape of the new one-hot axis.     |
+------------------+----------------------------------------------------------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | ``shape``                      |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} =
   \begin{cases}
   1&\text{if }i_{\mathtt{one\_hot\_axis}} = \mathtt{arg}_{(i : i\ne \mathtt{one\_hot\_axis})}\\
   0&\text{otherwise}
   \end{cases}

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::OneHot
   :project: ngraph
   :members:
