.. backend-support/index.rst


Transformer, CPU, GPU, PlaidML
###############################


* :ref:`hybrid_transformer`
* :ref:`cpu_backend`
* :ref:`plaidml_backend`
* :ref:`gpu_backend`




What is a backend?
------------------

Backends are responsible for function execution and value allocation. They 
can be used to :doc:`carry out a programmed computation<../howto/execute>`
from a framework by using a CPU or GPU; or they can be used with an *Interpreter* 
mode, which is primarily intended for testing, to analyze a program, or for a 
framework developer to develop customizations. Experimental APIs to support 
current and future nGraph Backends are also available; see, for example, the 
section on :ref:`plaidml_backend`.



.. _hybrid_transformer:

Hybrid Transformer
==================

Lorem ipsum


.. _cpu_backend:

CPU Backend
===========

Lorem ipsum


.. _gpu_backend:

GPU Backend
===========

Lorem ipsum


.. _plaidml_backend:

PlaidML Backend
===============

The nGraph ecosystem has recently added initial (experimental) support for `PlaidML`_,
which is an advanced :abbr:`Machine Learning (ML)` library that can further
accelerate training models built on GPUs. When you select the ``PlaidML`` option
as a backend, it behaves as an advanced tensor compiler that can further speed up
training with large data sets.

.. _PlaidML: https://github.com/plaidml
