.. backend-support/index.rst

About backends
##############

* :ref:`what_is_backend`
* :ref:`how_to_use`


.. _what_is_backend:

What's a backend?
-----------------

In the nGraph Compiler stack, what we call a *backend* is responsible for 
function execution and value allocation. A  backend can be used to 
:doc:`carry out a programmed computation<../core/constructing-graphs/execute>` 
from a framework on a CPU or GPU; or it can be used with an *Interpreter* mode, 
which is primarily intended for testing, to analyze a program, or to help a 
framework developer customize targeted solutions. Experimental APIs to support 
current and future nGraph Backends are also available; see, for example, the 
section on :doc:`plaidml-ng-api/index`.

.. csv-table::
   :header: "Backend", "Current nGraph support", "Future nGraph support"
   :widths: 35, 10, 10

   Intel® Architecture Processors (CPUs), Yes, Yes
   Intel® Nervana™ Neural Network Processor™ (NNPs), Yes, Yes
   NVIDIA\* CUDA (GPUs), Yes, Some 
   AMD\* GPUs, Yes, Some


.. _how_to_use:

How to use?
-----------

#. Create a ``Backend``; think of it as a compiler. 
#. A ``Backend`` can then produce an ``Executable`` by calling ``compile``. 
#. A single iteration of the executable is executed by calling the ``call``
   method on the ``Executable`` object.

.. figure:: ../graphics/execution-interface.png
   :width: 650px

   The execution interface for nGraph 


The nGraph execution API for ``Executable`` objects is a simple, five-method 
interface; each backend implements the following five functions:


* The ``create_tensor()`` method allows the bridge to create tensor objects 
  in host memory or an accelerator's memory.
* The ``write()`` and ``read()`` methods are used to transfer raw data into 
  and out of tensors that reside in off-host memory.
* The ``compile()`` method instructs the backend to prepare an nGraph function 
  for later execution.
* And, finally, the ``call()`` method is used to invoke an nGraph function 
  against a particular set of tensors.
