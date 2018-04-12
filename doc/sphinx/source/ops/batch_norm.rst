.. batch_norm.rst:

#########
BatchNorm
#########

.. code-block:: cpp

   BatchNorm  // Produces a normalized output


Description
===========

Produces a normalized output.

Version 1:

.. code-block:: cpp

   BatchNorm(epsilon, gamma, beta, input) -> normalized, mean, variance

Attributes
----------

+-----------------+-----------------+---------------------+
| Name            | Type            | Notes               |
+=================+=================+=====================+
| ``epsilon``     | ``double``      | Bias for variance.  |
+-----------------+-----------------+---------------------+


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``gamma``       | any                     | \(C\)                          |
+-----------------+-------------------------+--------------------------------+
| ``beta``        | same as ``gamma``       | \(C\)                          |
+-----------------+-------------------------+--------------------------------+
| ``input``       | same as ``gamma``       | \(\*, C, ...\)                 |
+-----------------+-------------------------+--------------------------------+

Outputs
-------
+---------------------+-------------------------+--------------------------------+
| Name                | Element Type            | Shape                          |
+=====================+=========================+================================+
| ``normalized``      | same as ``gamma``       | same as ``input``              |
+---------------------+-------------------------+--------------------------------+
| ``batch_mean``      | same as ``gamma``       | \(C\)                          |
+---------------------+-------------------------+--------------------------------+
| ``batch_variance``  | same as ``gamma``       | \(C\)                          |
+---------------------+-------------------------+--------------------------------+

The ``batch_mean`` and ``batch_variance`` are computed per-channel from ``input``
and will serve as :math:`\mu` and :math:`\nu` in the mathematical definition.

Version 2:

.. code-block:: cpp

   BatchNorm(epsilon, gamma, beta, input, mean, variance) -> normalized

Attributes
----------

+-----------------+-----------------+---------------------+
| Name            | Type            | Notes               |
+=================+=================+=====================+
| ``epsilon``     | ``double``      | Bias for variance.  |
+-----------------+-----------------+---------------------+

Inputs
------
+---------------------+-------------------------+--------------------------------+
| Name                | Element Type            | Shape                          |
+=====================+=========================+================================+
| ``gamma``           | any                     | \(C\)                          |
+---------------------+-------------------------+--------------------------------+
| ``beta``            | same as ``gamma``       | \(C\)                          |
+---------------------+-------------------------+--------------------------------+
| ``input``           | same as ``gamma``       | \(\*, C, ...\)                 |
+---------------------+-------------------------+--------------------------------+
| ``global_mean``     | same as ``gamma``       | \(C\)                          |
+---------------------+-------------------------+--------------------------------+
| ``global_variance`` | same as ``gamma``       | \(C\)                          |
+---------------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``normalized``  | same as ``gamma``       | same as ``input``              |
+-----------------+-------------------------+--------------------------------+

The ``global_mean`` and ``global_variance`` will serve as :math:`\mu` and :math:`\nu`
in the mathematical definition.


Mathematical Definition
=======================

The axes of the input fall into two categories, positional and
channel, with channel being axis 1. For each position, there are
:math:`C` channel values, each normalized independently.

Normalization of a channel sample is controlled by two values, the
mean :math:`\mu`, and the variance :math:`\nu`, and two scaling
attributes, :math:`\gamma` and :math:`\beta`.

.. math::

   y_c = \frac{x_c-\mu_c}{\sqrt{\nu_c+\epsilon}}\gamma_c+\beta_c

The mean and variance can be arguments or computed for each channel of
``input`` over the positional axes. When computed from ``input``, the
mean and variance per channel are available as outputs.

Backprop
========

C++ Interface
=============

.. doxygenclass:: ngraph::op::BatchNorm
   :project: ngraph
   :members:
