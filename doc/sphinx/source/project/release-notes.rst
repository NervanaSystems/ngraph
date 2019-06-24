.. project/release-notes.rst:

Release Notes
#############

nGraph is provided as source code, APIs, build scripts, and some binary formats 
for various Compiler stack configurations and use cases. 

This page includes additional documentation updates.

We are pleased to announce the release of version |version|-doc.


0.21-doc
--------

Documentation updates
~~~~~~~~~~~~~~~~~~~~~

Summary of documentation-related changes:

+ Update :doc:`doc-contributor-README` for new community-based contributions. 
+ Added instructions on how to test or display the installed nGraph version.
+ Added instructions on building nGraph bridge (ngraph-bridge).
+ Updated Backend Developer Guides and ToC structure.
+ Tested documentation build on Clear Linux OS; it works.
+ Fixed a few links and redirs affected by filename changes.
+ Some coding adjustments for options to render math symbols, so they can be 
  documented more clearly and without excessive JS (see replacements.txt).
+ Consistent filenaming on all BE indexes.
+ Remove deprecated TensorAPI.
+ 


Core updates for |version|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+ The offset argument in tensor reads and writes has been removed
+ Save/load API
+ More ONNX ops
+ Better tensor creation
+ More shape support
+ Provenance improvements
+ offset arg for tensor creation is deprecated
+ static linking support


.. important:: Pre-releases (``-rc-0.*``) have newer features, and are less stable.  


Changelog on Previous Releases
==============================

For downloads formatted as ``.zip`` and ``tar.gz``, see 
https://github.com/NervanaSystems/ngraph/releases.

0.20
----

+ Save/load API
+ More ONNX ops
+ Better tensor creation
+ More shape support
+ Provenance improvements


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
+ Pad op takes CoordinateDiff instead of Shape pad values to allow for negative 
  padding.


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

.. _0.20.0-rc.0: https://github.com/NervanaSystems/ngraph/releases/tag/v0.20.0-rc.0_
.. _0.19.0-rc.2: https://github.com/NervanaSystems/ngraph/releases/tag/v0.19.0-rc.2_
.. _0.18.1: https://github.com/NervanaSystems/ngraph/releases/tag/v0.18.1_
.. _0.17.0-rc.1: `https://github.com/NervanaSystems/ngraph/releases/tag/v0.17.0-rc.1
.. _0.16.0-rc.3: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.3
.. _0.16.0-rc.2: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.2
.. _0.16.0-rc.1: https://github.com/NervanaSystems/ngraph/releases/tag/v0.16.0-rc.1
