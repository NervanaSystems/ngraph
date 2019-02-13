.. backend-support/cpp-api.rst:

Backend APIs 
############

* :ref:`backend-api`
* :ref:`executable-api`
* :ref:`tensor-api`
* :ref:`hosttensor-api`
* :ref:`plaidml-ng-api`


As of version ``0.15``, there is a new backend API to work with functions that 
can be compiled as a runtime ``Executable``. Where previously ``Backend`` used a 
``shared_ptr<Function>`` as the handle passed to the ``call`` method to execute 
a compiled object, the addition of the ``shared_ptr<Executable>`` object has 
more direct methods to actions such as ``validate``, ``call``, ``get_performance_data``, and so on. This new API permits any executable to be saved or loaded *into* or 
*out of* storage and makes it easier to distinguish when a Function is compiled,
thus making the internals of the ``Backend`` and ``Executable`` easier to 
implement.  

How to use?
-----------

#. Create a ``Backend``; think of it as a compiler. 
#. A ``Backend`` can then produce an ``Executable`` by calling ``compile``. 
#. A single iteration of the executable is executed by calling the ``call``
   method on the ``Executable`` object.


.. _backend-api:

Backend
=======


.. figure:: ../graphics/backend-dgm.png
   :width: 650px

   Various backends are accessible via nGraph core APIs


.. doxygenclass:: ngraph::runtime::Backend
   :project: ngraph
   :members:


.. _executable-api:

Executable
==========


.. figure:: ../graphics/runtime-exec.png
   :width: 650px

   The ``compile`` function on an ``Executable`` has more direct methods to 
   actions such as ``validate``, ``call``, ``get_performance_data``, and so on. 


.. doxygenclass:: ngraph::runtime::Executable
   :project: ngraph
   :members: 


.. _tensor-api:

Tensor
======

.. doxygenclass:: ngraph::runtime::Tensor
   :project: ngraph
   :members:


.. _hosttensor-api:

HostTensor
==========

.. doxygenclass:: ngraph::runtime::HostTensor
   :project: ngraph
   :members:


.. _plaidml-ng-api:

PlaidML
=======

.. doxygenclass:: ngraph::runtime::plaidml::PlaidML_Backend
   :project: ngraph
   :members: