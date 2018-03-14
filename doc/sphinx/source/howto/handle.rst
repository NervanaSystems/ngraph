.. handle.rst:

#################
Handle an Export 
#################

.. contents::
   
Intel nGraph++ can be used to run a model or to interact with model data 
that has been exported from a framework. This documentation explains the various 
means that users have to work with such "exported" data through nGraph APIs.  

An exporting entity will export a model from a framework, "serialize" the model, 
and create a file that exposes the state of the model at a specific runtime. For 
users of the nGraph APIs, this means that data can be captured and operated upon 
at any point a serialized model can be created.    


From ONNX
---------

`ONNX`_ can be used to create ``.onnx``-formatted files representing models 
that have been exported from a framework.  These are exported to a serialized 
format, and are usually named ``<something>.onnx``. 

After first following an "exporting" `tutorial from ONNX`_, you should have an 
``.onnx`` file as serialized by ONNX.     

The ``.onnx`` file can then be loaded to the nGraph for handling as follows:   

For a simple model ``y = a + b`` stored in an ONNX file named 
``y_equals_a_plus_b.onnx.pb``, load it to nGraph using the following code. 


.. https://github.com/NervanaSystems/private-ngraph/tree/master/ngraph/frontends/onnx 

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 34-47

The ``transformer.computation`` line is what creates an executable version of 
the model.



.. From NNVM
   ----------

.. From XLA
   --------

.. etc, eof 







.. _ONNX: http://onnx.ai
.. _tutorial from ONNX: https://github.com/onnx/tutorials

