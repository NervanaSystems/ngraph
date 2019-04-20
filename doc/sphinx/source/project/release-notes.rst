.. ngraph/release-notes:

Release Notes
#############

The latest |version| download below

* `Format .zip`_ 
* `Format tar.gz`_ 

See also: https://github.com/NervanaSystems/ngraph/releases for previous versions. 


CHANGELOG |release|
===================


nGraph v0.17.0-rc.0
-------------------

+ Allow negative padding in more places
+ Add code generation for some quantized ops
+ Preliminary dynamic shape support
+ initial distributed ops

Recent API Changes
~~~~~~~~~~~~~~~~~~

+ Pad op takes CoordinateDiff instead of Shape pad values to allow for negative padding.


Changelog 
=========

nGraph v0.16.0-rc.3
-------------------

+ NodeInput and NodeOutput classes prepare for simplifications of Node
+ Test improvements
+ Additional quantization ops
+ Performance improvements
+ Fix memory leak
+ Concat optimization
+ Doc updates


.. _Format .zip: https://github.com/NervanaSystems/ngraph/archive/v0.17.0-rc.0.zip
.. _Format tar.gz: https://github.com/NervanaSystems/ngraph/archive/v0.17.0-rc.0.tar.gz