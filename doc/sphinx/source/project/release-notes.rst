.. project/release-notes.rst:

Release Notes
#############

nGraph is provided as source code, APIs, build scripts, and some binary formats for 
various Compiler stack configurations and use cases. 

All project source code and documentation are maintained in a GitHub* repository.

.. Begin release notes template
   This is the `Release Notes` template for latest nGraph Compiler stack release versioning 

We are pleased to announce the release of version |version|.


What's new?
-----------

Additional functionality included with this release:




What's updated?
---------------

The following sections provide detailed lists of major updates and removals by component:


Core
~~~~



Frameworks
~~~~~~~~~~



Backends
~~~~~~~~



Visualization Tools
~~~~~~~~~~~~~~~~~~~



Other
~~~~~


.. ----------------------------------------------------------------------------
   End release notes template 


For downloads formatted as ``.zip`` and ``tar.gz``, see https://github.com/NervanaSystems/ngraph/releases.

.. important:: Pre-releases (``-rc-0.*``) have newer features, and are less stable.  


Changelog on Previous Releases
==============================






0.19
----

**Download** `0.19.0-rc.2`_

+ More dynamic shape preparation
+ Distributed interface factored out
+ fp16 and bfloat16 types
+ codegen execution parameterized by context
+ NodeMap, NodeVector, ParameterVector, ResultVector now vectors
  
  - ``node_vector.hpp`` replaced by ``node.hpp``
  - ``op/parameter_vector.hpp`` replaced by ``op/parameter.hpp``
  - ``op/result_vector.hpp`` replaced by ``op/result.hpp``

+ Additional ONNX ops
+ Add graph visualization tools to doc
+ Update doxygen to be friendlier to frontends



0.18
----

**Download** `0.18.1`_


+ Python formatting issue
+ mkl-dnn work-around
+ Event tracing improvements
+ Gaussian error function
+ Begin tracking framework node names
+ ONNX quantization
+ More fusions


0.17
----

**Download** `0.17.0-rc.1`_

+ Allow negative padding in more places
+ Add code generation for some quantized ops
+ Preliminary dynamic shape support
+ initial distributed ops
+ Pad op takes CoordinateDiff instead of Shape pad values to allow for negative padding.


0.16
----

* **Download**: `0.16.0-rc.3`_
* **Download** `0.16.0-rc.2`_
* **Download** `0.16.0-rc.1`_


+ NodeInput and NodeOutput classes prepare for simplifications of Node
+ Test improvements
+ Additional quantization ops
+ Performance improvements
+ Fix memory leak
+ Concat optimization
+ Doc updates

.. _0.19.0-rc.2: https://github.com/NervanaSystems/ngraph/releases/tag/v0.19.0-rc.2_
.. _0.18.1: https://github.com/NervanaSystems/ngraph/releases/tag/v0.18.1_
.. _0.17.0-rc.1: `https://github.com/NervanaSystems/ngraph/releases/tag/v0.17.0-rc.1
.. _0.16.0-rc.3: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.3
.. _0.16.0-rc.2: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.2
.. _0.16.0-rc.1: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.1
