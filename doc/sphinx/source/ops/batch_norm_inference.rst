.. batch_norm_inference.rst:

##################
BatchNormInference
##################

.. code-block:: cpp

   BatchNormInference  // Adjust input for mean and variance


Description
===========



Inputs
------

+---------------------+-------------------------+------------------------------+
| Name                | Element Type            | Shape                        |
+=====================+=========================+==============================+
| ``input``           | real                    | :math:`(\bullet, C, \ldots)` |
+---------------------+-------------------------+------------------------------+
| ``gamma``           | same as ``input``       | :math:`(C)`                  |
+---------------------+-------------------------+------------------------------+
| ``beta``            | same as ``input``       | :math:`(C)`                  |
+---------------------+-------------------------+------------------------------+
| ``mean``            | same as ``input``       | :math:`(C)`                  |
+---------------------+-------------------------+------------------------------+
| ``variances``       | same as ``input``       | :math:`(C)`                  |
+---------------------+-------------------------+------------------------------+


Attributes
----------

+------------------+--------------------+--------------------------------------------------------+
| Name             | Type               | Notes                                                  |
+==================+====================+========================================================+
| ``epsilon``      | ``double``         | Small bias added to variance to avoid division by 0.   |
+------------------+--------------------+--------------------------------------------------------+

Outputs
-------

+---------------------+-------------------------+-----------------------------+
| Name                | Element Type            | Shape                       |
+=====================+=========================+=============================+
| ``normalized``      | same as ``gamma``       | Same as ``input``           |
+---------------------+-------------------------+-----------------------------+

Mathematical Definition
=======================

The axes of the input fall into two categories: positional and channel, with 
channel being axis 1. For each position, there are :math:`C` channel values, 
each normalized independently.

Normalization of a channel sample is controlled by two values:

*  the `mean` :math:`\mu`, and
   
*  the `variance` :math:`\sigma^2`; 

and by two scaling attributes: :math:`\gamma` and :math:`\beta`. 

.. math::

   \mathtt{normalized}_{\bullet, c, \ldots} = \frac{\mathtt{input}_{\bullet, c, \ldots}-\mu_c}{\sqrt{\sigma^2_c+\epsilon}}\gamma_c+\beta_c


C++ Interface
==============

.. doxygenclass:: ngraph::op::v0::BatchNormInference
   :project: ngraph
   :members:


