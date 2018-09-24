.. drop_out.rst:

#######
DropOut
#######

.. code-block:: cpp

   DropOut  // DropOut operations

Description
===========

.. TODO

Inputs
------

.. TODO

+-----------------+-------------------------+----------------------------------+
| Name            | Element Type            | Shape                            |
+=================+=========================+==================================+
| ``arg``         | any                     | :math:`(N, C, d_1, \ldots, d_n)` |
+-----------------+-------------------------+----------------------------------+

Attributes
----------

.. TODO

+-------------------------------+-----------------------------------------------+
| Name                          | Description                                   |
+===============================+===============================================+
| ``window_shape``              | The window shape.                             |
+-------------------------------+-----------------------------------------------+
| ``window_movement_strides``   | The window movement strides. (defaults to 1s) |
+-------------------------------+-----------------------------------------------+
| ``padding_below``             | The below-padding shape. (defaults to 0s)     |
+-------------------------------+-----------------------------------------------+
| ``padding_above``             | The above-padding shape. (defaults to 0s)     |
+-------------------------------+-----------------------------------------------+

Outputs
-------

.. TODO

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | same as ``arg``         | :math:`(N,C,d'_1,\ldots,d'_n)` |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. TODO update this

Given an input data batch tensor :math:`T_{in}`, the output tensor is defined by the equation


.. math::
        T_{out}[a,c,i_1,\dots,i_n] =
	\max_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} (T_{in}[a,c,j_1,\dots,j_n])


C++ Interface
=============