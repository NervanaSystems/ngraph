.. ngraph/release-notes:

Release Notes
#############



|version|
|release|

For downloads formatted as ``.zip`` and ``tar.gz``, see https://github.com/NervanaSystems/ngraph/releases; 

.. important:: Pre-releases (``-rc-0.*``) have newer features, and are less stable.  

CHANGELOG |release|
===================

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


Changelog 
=========


nGraph v0.18.1
--------------

+ Python formatting issue
+ mkl-dnn work-around
+ Event tracing improvements
+ Gaussian error function
+ Begin tracking framework node names
+ ONNX quantization
+ More fusions


nGraph v0.17.0-rc.0
-------------------

+ Allow negative padding in more places
+ Add code generation for some quantized ops
+ Preliminary dynamic shape support
+ initial distributed ops

Recent API Changes
~~~~~~~~~~~~~~~~~~

+ Pad op takes CoordinateDiff instead of Shape pad values to allow for negative padding.


nGraph v0.16.0-rc.3
-------------------

+ NodeInput and NodeOutput classes prepare for simplifications of Node
+ Test improvements
+ Additional quantization ops
+ Performance improvements
+ Fix memory leak
+ Concat optimization
+ Doc updates
