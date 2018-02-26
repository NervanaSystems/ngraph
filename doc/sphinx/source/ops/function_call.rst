.. function_call.rst:

############
FunctionCall
############

.. code-block:: cpp

   FunctionCall  // Function call operation


Description
===========

Calls the specified function on ``args``. The results of the function are the outputs
of the op.

Inputs
------

+------------+--------------------+----------------------------------------------+
| Name       | Type               |                                              |
+============+====================+==============================================+
| ``args``   | ``ngraph::Nodes``  | Element types and shapes must correspond to  |
|            |                    | the parameters of ``function``               |
+------------+--------------------+----------------------------------------------+

Attributes
----------

+----------------+---------------------------------------+
| Name           | Type                                  |
+================+=======================================+
| ``function``   | ``std::shared_ptr<ngraph::Function>`` |
+----------------+---------------------------------------+

Outputs
-------

One output for each function result.

C++ Interface
=============

.. doxygenclass:: ngraph::op::FunctionCall
   :project: ngraph
   :members:

