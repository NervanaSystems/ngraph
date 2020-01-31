.. convolution.rst:

###########
Convolution
###########

.. code-block:: cpp

   Convolution  // A batched convolution operation


Description
===========

.. TODO 

Long description



Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``image_batch`` | Any                     | ``(N, C_in, d_1, ..., d_n)``   |
+-----------------+-------------------------+--------------------------------+
| ``filters``     | Same as ``image_batch`` | ``(N, C_in, df_1, ..., df_n)`` |
+-----------------+-------------------------+--------------------------------+

Attributes
----------

+-----------------------------+-----------------------------+----------------------------------------+
| Name                        | Type                        | Notes                                  |
+=============================+=============================+========================================+
| ``window_movement_strides`` | ``Strides[n]``              | How far to slide the                   |
|                             |                             | window along each axis at each step    |
+-----------------------------+-----------------------------+----------------------------------------+
| ``window_dilation_strides`` | ``Strides[n]``              | Per-axis dilation to apply to the      |
|                             |                             | filters                                |
+-----------------------------+-----------------------------+----------------------------------------+
| ``padding_below``           | ``Shape[n]``                | How many padding elements to add       |
|                             |                             | below the 0-coordinate on each axis    |
+-----------------------------+-----------------------------+----------------------------------------+
| ``padding_above``           | ``Shape[n]``                | How many padding elements to add above |
|                             |                             | the max-coordinate on each axis        |
+-----------------------------+-----------------------------+----------------------------------------+
| ``image_dilation_strides``  | ``Strides[n]``              | Per-axis dilation to apply to the      |
|                             |                             | image batch                            |
+-----------------------------+-----------------------------+----------------------------------------+


Outputs
-------

+------------------+-------------------------+----------------------------------------------------+
| Name             | Element Type            | Shape                                              |
+==================+=========================+====================================================+
| ``features_out`` | Same as ``image_batch`` | ``(N, C_in, d_1 - df_1 + 1, ..., d_n - df_n + 1)`` |
+------------------+-------------------------+----------------------------------------------------+

It must be the case that after dilation and padding are applied, the filter fits within the image.

.. TODO image add


Mathematical Definition
=======================

Padding
-------

Let :math:`p` (the padding below) and :math:`q` (the padding above) be a sequence of :math:`n`
integers, and :math:`T` be a tensor of shape :math:`(d_1,\dots,d_n)`, such that for all :math:`i`,
:math:`p_i + d_i + q_i \ge 0`. Then :math:`\mathit{Pad}[p,q](T)` is the tensor of shape
:math:`(p_1 + d_1 + q_1,\dots,p_n + d_n + q_n)` such that

.. math::

   \mathit{Pad}[p,q](T)_{i_1,\dots,i_n} \triangleq \begin{cases}
                                                      T_{i_1 - p_1,\dots,i_n - p_n} &\mbox{if for all }j, i_j \ge p_j\mbox{ and }i_j < p_j + d_j \\
                                                      0                             &\mbox{otherwise.}
                                                   \end{cases}

Dilation
--------

Let :math:`l` (the dilation strides) be a sequence of :math:`n` positive integers, and :math:`T`
be a tensor of shape :math:`(d_1,\dots,d_n)`. Then :math:`\mathit{Dilate}[l](T)` is the tensor of
shape :math:`(d'_1,\dots,d'_n)` where :math:`d'_i = \mathit{max}(0,l_i(d_i - 1) + 1)` such that

.. math::

   \mathit{Dilate}[l](T)_{i_1,\dots,i_n} \triangleq \begin{cases}
                                                       T_{i_1/l_1,\dots,i_n/l_n} &\mbox{if for all }j, i_j\mbox{ is a multiple of }l_j \\
                                                       0                         &\mbox{otherwise.}
                                                    \end{cases}

Striding
--------

Let :math:`s` (the strides) be a sequence of :math:`n` positive integers, and :math:`T` be a
tensor of shape :math:`(d_1,\dots,d_n)`. Then :math:`\mathit{Stride}[s](T)` is the tensor of
shape :math:`(d'_1,\dots,d'_n)` where :math:`d'_i = \left\lceil \frac{d_i}{s_i} \right\rceil`
such that

.. math::

   \mathit{Stride}[s](T)_{i_1,\dots,i_n} \triangleq T_{s_1i_1,\dots,s_ni_n}


:math:`s` is the how far, not the unit of farness.

Convolution
-----------

.. image possibly imported soon; they are not big files but they are svg 

.. 
 figure:: ../graphics/classngraph_1_1op_1_1Convolution__coll__graph_org.svg
 :height: 500px


Padded, Dilated, Strided Convolution
------------------------------------

.. math::

   \mathit{PDSConv}[g,p,q,l,s](T_\mathit{image},T_\mathit{filter} \triangleq \mathit{Stride}[s](\mathit{Conv}(\mathit{Pad}[p,q](\mathit{Dilate}[g](T_\mathit{batch})),\mathit{Dilate}[l](T_\mathit{filter})))

Batched, Padded, Dilated, Strided Convolution
---------------------------------------------

.. TODO

C++ Interface
=============

  .. doxygenclass:: ngraph::op::v0::Convolution
     :project: ngraph
     :members:

     