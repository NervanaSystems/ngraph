.. index.rst


#######################
Interact with Backends 
#######################



Backend
=======

Backends are responsible for function execution and value allocation. They 
can be used to :doc:`carry out a programmed computation<../howto/execute>`
from a framework by using a CPU or GPU; or they can be used with the ``INT`` or
*Interpreter* mode, which is primarily intended for testing, to analyze a
program, or for a framework developer to develop a custom UI or API.


.. figure:: ../graphics/backends_inhdgm.png
   :width: 553px


.. doxygenclass:: ngraph::runtime::Backend
   :project: ngraph
   :members:


TensorView
==========

.. figure:: ../graphics/tensorview_inhdgm.png
   :width: 553px

Tensor

.. doxygenclass:: ngraph::runtime::Tensor
   :project: ngraph
   :members:


PlaidML
=======

The nGraph ecosystem has recently added initial (experimental) support for `PlaidML`_,
which is an advanced :abbr:`Machine Learning (ML)` library that can further
accelerate training models built on GPUs. When you select the ``PlaidML`` option
as a backend, it behaves as an advanced tensor compiler that can further speed-up
training with large data sets.


.. figure:: ../graphics/plaidml-plaidml.png
   :width: 148px

.. doxygenclass:: ngraph::runtime::plaidml::PlaidML_Backend
   :project: ngraph
   :members:



.. _PlaidML: https://github.com/plaidml
