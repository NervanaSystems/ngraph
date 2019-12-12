.. batch_norm_training.rst:

#################
BatchNormTraining
#################

.. code-block:: cpp

   BatchNormTraining  // Compute mean and variance from the input.


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
| ``batch_mean``      | same as ``gamma``       | :math:`(C)`                 |
+---------------------+-------------------------+-----------------------------+
| ``batch_variance``  | same as ``gamma``       | :math:`(C)`                 |
+---------------------+-------------------------+-----------------------------+

The ``batch_mean`` and ``batch_variance`` outputs are computed per-channel from 
``input``.


Mathematical Definition
=======================

The axes of the input fall into two categories: positional and channel, with 
channel being axis 1. For each position, there are :math:`C` channel values, 
each normalized independently.

Normalization of a channel sample is controlled by two values:

*  the `batch_mean` :math:`\mu`, and
   
*  the `batch_variance` :math:`\sigma^2`; 

and by two scaling attributes: :math:`\gamma` and :math:`\beta`. 

The values for :math:`\mu` and :math:`\sigma^2` come from computing the 
mean and variance of ``input``.

.. math::

   \mu_c &= \mathop{\mathbb{E}}\left(\mathtt{input}_{\bullet, c, \ldots}\right)\\
   \sigma^2_c &= \mathop{\mathtt{Var}}\left(\mathtt{input}_{\bullet, c, \ldots}\right)\\
   \mathtt{normlized}_{\bullet, c, \ldots} &= \frac{\mathtt{input}_{\bullet, c, \ldots}-\mu_c}{\sqrt{\sigma^2_c+\epsilon}}\gamma_c+\beta_c

Backprop
========

.. math::

   [\overline{\texttt{input}}, \overline{\texttt{gamma}}, \overline{\texttt{beta}}]=\\
   \mathop{\texttt{BatchNormTrainingBackprop}}(\texttt{input},\texttt{gamma},\texttt{beta},\texttt{mean},\texttt{variance},\overline{\texttt{normed_input}}).



C++ Interface
==============

.. doxygenclass:: ngraph::op::v0::BatchNormTraining
   :project: ngraph
   :members:


