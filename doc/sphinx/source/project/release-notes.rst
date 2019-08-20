.. project/release-notes.rst:

:orphan:

Release Notes
#############

nGraph is provided as source code, APIs, build scripts, and some binary formats 
for various Compiler stack configurations and use cases. 

For downloads formatted as ``.zip`` and ``tar.gz``, see 
https://github.com/NervanaSystems/ngraph/releases.

This page includes additional documentation updates.

We are pleased to announce the release of version |version|.


Core updates for |version|
--------------------------

Allow DLLs that link nGraph statically to load backends


.. important:: Pre-releases (``-rc-0.*``) have newer features, and are less stable.  


Changelog on Previous Releases
==============================


0.25.0
------

+ Better PlaidML support
+ Double-buffering support
+ Constant folding
+ Support for static linking
+ Additional ops
+ Preliminary static linking support
+ Known issue: No PlaidML training support
+ Doc: Add instructions how to build NGRAPH_PLAIDML backend
+ Published interim version of doc navigation for updates at ngraph.ai
+ GPU validations: added 5 functional TensorFlow workloads and 4 functional 
  ONNX workloads


0.24
----

+ Fixes reshape sink/swim issue
+ More ONNX ops
+ Elementwise divide defaults to Python semantics
+ GenerateMask seed optional
+ Graph visualization improvements
+ Preserve control dependencies in more places
+ GetOutputElement has single input


0.23
----

+ More ONNX ops
+ Elementwise divide defaults to Python semantics
+ GenerateMask seed optional
+ Document new debug tool
+ Graph visualization improvements
+ Note deprecation of MXNet's ``ngraph-mxnet`` PyPI
+ Note default change to `svg` files for graphs and visualization
+ Add more prominent tips for contributors who find the doc-contributor-README
+ Better GSG / Install Guide structure.
+ Added group edits and new illustrations from PR 2994 to `introduction.rst`.
+ Ensure ngraph-bridge link in README goes to right place.
+ Make project `extras` their own subdirectory with index to help organize them.
+ **Known Issues**
  
  - When using TensorFlow\* v1.14.0 with ```ngraph-bridge`` v0.16.0rc0 and CPU
    backend, we saw notable to severe decreases in throughput in many models.

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
+ Updated ``doc-contributor-README`` for new community-based contributions. 
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
