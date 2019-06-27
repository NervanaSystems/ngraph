.. backends/backend-api/index.rst:


Backend
=======

:ref:`more_resources`

.. doxygenclass:: ngraph::runtime::Backend
   :project: ngraph
   :members:



.. _more_resources: 

Additional resources
====================

OpenCL
------

#. Install the latest Linux driver for your system from 
   https://software.intel.com/en-us/articles/opencl-drivers;
   You may need to install `OpenCL SDK`_ in case of an 
   libOpenCL.so absence.

#. Any user added to "video" group: 

   .. code-block:: console 

      sudo usermod –a –G video <user_id>

   may, for example, be able to find details at the
   ``/sys/module/[system]/parameters/`` location. 



nGraph Bridge 
~~~~~~~~~~~~~

When specified as the generic backend -- either 
manually or automatically from a framework --  
``NGRAPH`` defaults to CPU, and it also allows 
for additional device configuration or selection. 

Because nGraph can select backends, you may try 
specifying the ``INTELGPU`` backend as a runtime 
environment variable: 

:envvar:`NGRAPH_TF_BACKEND="INTELGPU"`

An `axpy.py example`_ is optionally available to test;
outputs will vary depending on the parameters
specified. 

.. code-block:: console

   NGRAPH_TF_BACKEND="INTELGPU" python3 axpy.py

* ``NGRAPH_INTELGPU_DUMP_FUNCTION`` -- dumps 
  nGraph’s functions in dot format.

* `` `` --.

* `` `` --.

* `` `` --.

.. _axpy.py example: https://github.com/tensorflow/ngraph-bridge/blob/master/examples/axpy.py
.. _OpenCL SDK: https://software.intel.com/en-us/opencl-sdk