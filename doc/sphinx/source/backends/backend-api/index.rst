.. backends/backend-api/index.rst:


.. _backend_api_macros:

Backend APIs
############

Each backend ``BACKEND`` needs to define the macro ``${BACKEND}_API`` 
appropriately to import symbols referenced from outside the library, 
and to export them from within the library. See any of the
``${backend}_backend_visibility`` header files for an example; see 
also :ref:`what_is_backend`

.. 


.. doxygenclass:: ngraph::runtime::Backend
   :project: ngraph
   :members:
