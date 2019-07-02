.. backends/executable-api/index.rst:


Executable
==========

* :ref:`generic_executable`
* :ref:`intelgpu_executable`


The ``compile`` function on an ``Executable`` has more direct methods to 
actions such as ``validate``, ``call``, ``get_performance_data``, and so on. 


.. _generic_executable:

Generic executable
------------------

.. doxygenclass:: ngraph::runtime::Executable
   :project: ngraph
   :members: 


.. _intelgpu_executable:

IntelGPU executable
-------------------

.. doxygenclass:: ngraph::runtime::intelgpu::IntelGPUExecutable
   :project: ngraph
   :members: 
