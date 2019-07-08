.. project/release-notes.rst:

Release Notes
#############

nGraph is provided as source code, APIs, build scripts, and some binary formats 
for various Compiler stack configurations and use cases. 

For downloads formatted as ``.zip`` and ``tar.gz``, see 
https://github.com/NervanaSystems/ngraph/releases.

This page includes additional documentation updates.

We are pleased to announce the release of version |version|-doc.


Core updates for |version|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+ PlaidML support
+ More ONNX ops
+ Elementwise divide defaults to Python semantics
+ GenerateMask seed optional


Latest doc updates
~~~~~~~~~~~~~~~~~~

+ Document new debug tool
+ Note deprecation of MXNet's ``ngraph-mxnet`` PyPI
+ Note default change to `svg` files for graphs and visualization
+ Add more prominent tips for contributors who find the doc-contributor-README


.. important:: Pre-releases (``-rc-0.*``) have newer features, and are less stable.  


Changelog on Previous Releases
==============================

0.22
----

+ More ONNX ops
+ Optimizations
+ Don't reseed RNG on each use
+ Initial doc and API for IntelGPU backend 
+ DynamicBackend API


0.21
----

+ The offset argument in tensor reads and writes has been removed
+ Save/load API
+ More ONNX ops
+ Better tensor creation
+ More shape support
+ Provenance improvements
+ offset arg for tensor creation is deprecated
+ static linking support
+ Initial test of 0.21-doc
+ Updated :doc:`doc-contributor-README` for new community-based contributions. 
+ Added instructions on how to test or display the installed nGraph version.
+ Added instructions on building nGraph bridge (ngraph-bridge).
+ Updated Backend Developer Guides and ToC structure.
+ Tested documentation build on Clear Linux OS; it works.
+ Fixed a few links and redirs affected by filename changes.
+ Some coding adjustments for options to render math symbols, so they can be 
  documented more clearly and without excessive JS (see replacements.txt).
+ Consistent filenaming on all BE indexes.
+ Removed deprecated TensorAPI.


0.20
----

+ Save/load API
+ More ONNX ops
+ Better tensor creation
+ More shape support
+ Provenance improvements


0.19
----

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

+ Python formatting issue
+ mkl-dnn work-around
+ Event tracing improvements
+ Gaussian error function
+ Begin tracking framework node names
+ ONNX quantization
+ More fusions


0.17
----

+ Allow negative padding in more places
+ Add code generation for some quantized ops
+ Preliminary dynamic shape support
+ initial distributed ops
+ Pad op takes CoordinateDiff instead of Shape pad values to allow for negative 
  padding.


0.16
----

+ NodeInput and NodeOutput classes prepare for simplifications of Node
+ Test improvements
+ Additional quantization ops
+ Performance improvements
+ Fix memory leak
+ Concat optimization
+ Doc updates
