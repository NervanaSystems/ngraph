.. core/overview.rst:


Overview
========

What follows here is a table of all documented namespaces with brief 
descriptions:


Namespace List
--------------
 
.. csv-table::
   :header: "Namespace", "Description", "Location in Repo", "Docs"
   :widths: 23, 53, 13, 23
   :escape: ~

   ``ngraph``, The Intel nGraph C++ API, `Nngraph`_, Implicit namespace omitted from most API documentation
   ``builder``, "Convenience functions that create additional graph nodes to implement commonly-used recipes; for example, auto-broadcast", `Nbuilder`_, Coming Soon
   ``descriptor``, Descriptors are compile-time representations of objects that will appear at run-time, `Ndescriptor`_, Coming Soon
   ``op``, Ops used in graph construction, `Nop`_, :doc:`../ops/index`
   ``runtime``, The objects and methods used for executing the graph, `Nruntime`_, :doc:`../backend-support/cpp-api`


.. _Nngraph: https://github.com/NervanaSystems/ngraph/tree/master/src/ngraph
.. _Nbuilder: https://github.com/NervanaSystems/ngraph/tree/master/src/ngraph/builder
.. _Ndescriptor: https://github.com/NervanaSystems/ngraph/tree/master/src/ngraph/descriptor
.. _Nop: https://github.com/NervanaSystems/ngraph/tree/master/src/ngraph/op
.. _Nruntime: https://github.com/NervanaSystems/ngraph/tree/master/src/ngraph/runtime




