.. index.rst


###################################
Interact with Programmable Backends 
###################################

Backend
========

Backends are responsible for function execution and value allocation. They 
can be used to :doc:`carry out a programmed computation<../howto/execute>`,
such as with a CPU or GPU; or they can be used with an *Interpreter* mode,
such as with an FPGA-enabled CPU to write programs that can or will make 
use of solutions involving discrete FPGAs.


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



