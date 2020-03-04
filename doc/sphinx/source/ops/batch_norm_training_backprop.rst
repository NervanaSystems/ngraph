.. batch_norm_training_backprop.rst:

#########################
BatchNormTrainingBackprop
#########################

.. code-block:: cpp

   BatchNormTrainingBackprop  // Compute mean and variance backprop from the input.


Description
===========

Computes the ``input``, ``gamma`` and ``beta`` backprop increments.


Inputs
------

+----------------------+-------------------------+------------------------------+
| Name                 | Element Type            | Shape                        |
+======================+=========================+==============================+
| ``input``            | real                    | :math:`(\bullet, C, \ldots)` |
+----------------------+-------------------------+------------------------------+
| ``gamma``            | same as ``input``       | :math:`(C)`                  |
+----------------------+-------------------------+------------------------------+
| ``beta``             | same as ``input``       | :math:`(C)`                  |
+----------------------+-------------------------+------------------------------+
| ``mean``             | same as ``input``       | :math:`(C)`                  |
+----------------------+-------------------------+------------------------------+
| ``variance``         | same as ``input``       | :math:`(C)`                  |
+----------------------+-------------------------+------------------------------+
| ``normalized_delta`` | same as ``input``       | same as ``input``            |
+----------------------+-------------------------+------------------------------+


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
| ``input_delta``     | same as ``input``       | Same as ``input``           |
+---------------------+-------------------------+-----------------------------+
| ``gamma_delta``     | same as ``gamma``       | :math:`(C)`                 |
+---------------------+-------------------------+-----------------------------+
| ``beta_delta``      | same as ``beta``        | :math:`(C)`                 |
+---------------------+-------------------------+-----------------------------+


Mathematical Definition
=======================

It is easiest to simplify by looking at a single channel and flattening the
remaining axes into a vector; so ``gamma`` and ``beta`` are scalars, and ``input`` is an
:math:`N`-element vector.

The step by step forward training computation is

.. math::
   
   \mathtt{mean} &= \frac{\sum{\mathtt{input}_i}}{N}\\
   \mathtt{centered}_i &= \mathtt{input}_i - \mathtt{mean}\\
   \mathtt{square}_i &= \mathtt{centered}_i^2\\
   \mathtt{variance} &= \frac{\sum \mathtt{square}_i}{N}\\
   \mathtt{invsqrt} &= \frac{1}{\sqrt{\mathtt{variance}+\epsilon}}\\
   \mathtt{gmul} &= \texttt{gamma}\cdot \mathtt{invsqrt}\\
   \mathtt{normed}_i &= \mathtt{centered}_i\mathtt{gmul}+\texttt{beta}

Using the notation :math:`\overline{\texttt{name}}` for :math:`\texttt{name_delta}`
and :math:`\overline{x} \leftarrow y`
to mean the backprop value for :math:`\texttt{x_delta}` is a sum that includes :math:`y`.

We work backwards

.. math::

   \overline{\texttt{beta}}&\leftarrow \overline{\texttt{normed}}\\
   \overline{\texttt{gmul}}&\leftarrow \sum \overline{\texttt{normed}}_i\\
   \overline{\texttt{centered}}_i&\leftarrow\overline{\texttt{normed}}_i\texttt{gmul}\\
   \overline{\texttt{gamma}}&\leftarrow \overline{\texttt{gmul}}\cdot\texttt{invsqrt}\\
   \overline{\texttt{invsqrt}}&\leftarrow\texttt{gamma}\cdot\overline{\texttt{gmul}}\\
   \overline{\texttt{variance}}&\leftarrow -\frac{\overline{\texttt{invsqrt}}\cdot\texttt{invsqrt}}{2\cdot(\texttt{variance}+\epsilon)}\\
   \overline{\texttt{square}}_i&\leftarrow\frac{\overline{\texttt{variance}}}{N}\\
   \overline{\texttt{centered}}_i&\leftarrow 2\cdot\texttt{centered}_i\cdot\overline{\texttt{square}}_i\\
   \overline{\texttt{input}}_i&\leftarrow\overline{\texttt{centered}}_i\\
   \overline{\texttt{mean}}&\leftarrow\sum\overline{\texttt{centered}}_i\\
   \overline{\texttt{input}}_i&\leftarrow\frac{\overline{\texttt{mean}}}{N}


C++ Interface
==============

.. doxygenclass:: ngraph::op::v0::BatchNormTrainingBackprop
   :project: ngraph
   :members:


