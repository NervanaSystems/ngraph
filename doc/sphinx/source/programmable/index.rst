.. index.rst


#######################
Interact with Backends 
#######################

Backend
========

Backends are responsible for function execution and value allocation. They 
can be used to :doc:`carry out a programmed computation<../howto/execute>`,
from a framework by using a CPU or GPU; or they can be used with an *Interpreter* 
mode, which is primary intended for testing, to analyze a program, or to be 
help a framework developer create a custom UI for runtime options in their 
framework. 


.. figure:: ../graphics/runtime.png
   :width: 650px


.. doxygenclass:: ngraph::runtime::Backend
   :project: ngraph
   :members:




TensorView
===========

.. doxygenclass:: ngraph::runtime::TensorView
   :project: ngraph
   :members:



