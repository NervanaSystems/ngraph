.. backend-support/index.rst


About backends
##############

* :ref:`what_is_backend`
* :ref:`hybrid_transformer`
* :ref:`cpu_backend`
* :ref:`plaidml_backend`
* :ref:`gpu_backend`


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
section on the :ref:`plaidml_backend`.


.. _hybrid_transformer:

Hybrid Transformer
==================

More detail coming soon


.. _cpu_backend:

CPU Backend
===========

More detail coming soon


.. _gpu_backend:

GPU Backend
===========

More detail coming soon 


.. _plaidml_backend:

PlaidML Backend
===============

The nGraph ecosystem has recently added initial (experimental) support for `PlaidML`_,
which is an advanced :abbr:`Machine Learning (ML)` library that can further
accelerate training models built on GPUs. When you select the ``PlaidML`` option
as a backend, it behaves as an advanced tensor compiler that can further speed up
training with large data sets.

.. _PlaidML: https://github.com/plaidml
