nGraph Compiler stack
~~~~~~~~~~~~~~~~~~~~~

https://github.com/NervanaSystems/ngraph[nGraph] is an open-source graph
compiler for Artificial Neural Networks (ANNs). The nGraph Compiler
stack provides an inherently efficient graph-based compilation
infrastructure designed to be compatible with the many of the upcoming
ASICs, like the Intel® Nervana™ Neural Network Processors (Intel®
Nervana™ NNPs), while also unlocking a massive performance boost on any
existing hardware targets in your neural network: both GPUs and CPUs.
Using its flexible infrastructure, you will find it becomes much easier
to create Deep Learning (DL) models that can adhere to the ``write once,
run anywhere'' mantra that enables your AI solutions to easily go from
concept to production to scale.

Frameworks using nGraph to execute workloads have shown
https://ai.intel.com/ngraph-compiler-stack-beta-release/[up to 45X]
performance boost compared to native implementations.

Using the Python API
^^^^^^^^^^^^^^^^^^^^

nGraph can be used directly with the
https://ngraph.nervanasys.com/docs/latest/python_api/[Python API]
described here, or with the
https://ngraph.nervanasys.com/docs/latest/backend-support/cpp-api.html[C++
API] described in the
https://ngraph.nervanasys.com/docs/latest/core/overview.html[core
documentation]. Alternatively, its performance benefits can be realized
through a frontend such as
https://pypi.org/project/ngraph-tensorflow-bridge/[TensorFlow],
https://pypi.org/project/ngraph-mxnet/[MXNet], and
https://pypi.org/project/ngraph-onnx/[ONNX]. You can also create your
own custom framework to integrate directly with the
http://ngraph.nervanasys.com/docs/latest/ops/index.html[nGraph Ops] for
highly-targeted graph execution.

Installation
~~~~~~~~~~~~

nGraph is available as binary wheels you can install from PyPI. nGraph
binary wheels are currently tested on Ubuntu 16.04 and require a CPU
with Intel® Advanced Vector Extensions 512 (Intel® AVX-512)
instructions. For other systems, you may want to
https://github.com/NervanaSystems/ngraph/blob/master/python/BUILDING.md[build]
from sources.

Installing nGraph Python API from PyPI is easy:

....
pip install ngraph-core
....

Usage example
~~~~~~~~~~~~~

Using nGraph’s Python API to construct a computation graph and execute a
computation is simple. The following example shows how to create a
minimal `(A + B) * C` computation graph and calculate a result using 3
numpy arrays as input.

[source,python]
----
import numpy as np
import ngraph as ng

A = ng.parameter(shape=[2, 2], name='A', dtype=np.float32)
B = ng.parameter(shape=[2, 2], name='B', dtype=np.float32)
C = ng.parameter(shape=[2, 2], name='C', dtype=np.float32)
# >>> print(A)
# <Parameter: 'A' ([2, 2], float)>

model = (A + B) * C
# >>> print(model)
# <Multiply: 'Multiply_14' ([2, 2])>

runtime = ng.runtime(backend_name='CPU')
# >>> print(runtime)
# <Runtime: Backend='CPU'>

computation = runtime.computation(model, A, B, C)
# >>> print(computation)
# <Computation: Multiply_14(A, B, C)>

value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
value_c = np.array([[9, 10], [11, 12]], dtype=np.float32)

result = computation(value_a, value_b, value_c)
# >>> print(result)
# [[ 54.  80.]
#  [110. 144.]]

print('Result = ', result)
----
