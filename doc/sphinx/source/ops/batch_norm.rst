.. batch_norm.rst:

#########
BatchNorm
#########

.. code-block:: cpp

   BatchNorm  // Produces a normalized output


Description
===========

Produces a normalized output.

Inputs
------

+---------------------+-------------------------+-----------------------------+
| Name                | Element Type            | Shape                       |
+=====================+=========================+=============================+
| ``input``           | same as ``gamma``       | \(..., C, ...\)             |
+---------------------+-------------------------+-----------------------------+
| ``gamma``           | any                     | \(C\)                       |
+---------------------+-------------------------+-----------------------------+
| ``beta``            | same as ``gamma``       | \(C\)                       |
+---------------------+-------------------------+-----------------------------+
| ``global_mean``     | same as ``gamma``       | \(C\)                       |
+---------------------+-------------------------+-----------------------------+
| ``global_variance`` | same as ``gamma``       | \(C\)                       |
+---------------------+-------------------------+-----------------------------+
| ``use_global``      | ``bool``                | \(\)                        |
+---------------------+-------------------------+-----------------------------+


Attributes
----------

+------------------+--------------------+---------------------+
| Name             | Type               | Notes               |
+==================+====================+=====================+
| ``epsilon``      | same as ``input``  | Bias for variance   |
+------------------+--------------------+---------------------+
| ``channel_axis`` | size_t             | Channel axis        |
+------------------+--------------------+---------------------+

Outputs
-------

+---------------------+-------------------------+-----------------------------+
| Name                | Element Type            | Shape                       |
+=====================+=========================+=============================+
| ``normalized``      | same as ``gamma``       | same as ``input``           |
+---------------------+-------------------------+-----------------------------+
| ``batch_mean``      | same as ``gamma``       | \(C\)                       |
+---------------------+-------------------------+-----------------------------+
| ``batch_variance``  | same as ``gamma``       | \(C\)                       |
+---------------------+-------------------------+-----------------------------+

The ``batch_mean`` and ``batch_variance`` outputs are computed per-channel from 
``input``. The values only need to be computed if ``use_global`` is ``false``, 
or if they are used.


Mathematical Definition
=======================

The axes of the input fall into two categories: positional and channel, with 
channel being axis 1. For each position, there are :math:`C` channel values, 
each normalized independently.

Normalization of a channel sample is controlled by two values:

*  the mean :math:`\mu`, and 
*  the variance :math:`\sigma^2`; 

and by two scaling attributes: :math:`\gamma` and :math:`\beta`. 

The values for :math:`\mu` and :math:`\sigma^2` come either from computing the 
mean and variance of ``input``, or from ``global_mean`` and ``global_variance``, 
depending on the value of ``use_global``.

.. math::

   y_c = \frac{x_c-\mu_c}{\sqrt{\sigma^2_c+\epsilon}}\gamma_c+\beta_c

The mean and variance can be arguments, or they may be computed for each channel 
of ``input`` over the positional axes. When computed from ``input``, the mean 
and variance per-channel are available as outputs.


C++ Interface
==============

.. doxygenclass:: ngraph::op::BatchNormTraining
   :project: ngraph
   :members:


.. doxygenclass:: ngraph::op::BatchNormInference
   :project: ngraph
   :members:


