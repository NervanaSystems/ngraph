## nGraph Neural Network compiler

[nGraph][ngraph_github] is Intel's open-source graph compiler and runtime for Neural Network models. Frameworks using nGraph to execute workloads have shown up to [45X](https://ai.intel.com/ngraph-compiler-stack-beta-release/) performance boost compared to native implementations.

nGraph can be used directly thought it's [Python API][api_python] or [C++ API][api_cpp]. Alternatively it can be used through one of its frontends, e.g. [TensorFlow][frontend_tf], [MXNet][frontend_mxnet] and [ONNX][frontend_onnx].

## Installation

nGraph is available as binary wheels you can install from PyPI. nGraph binary wheels are currently tested on Ubuntu 16.04 and require a CPU with AVX-512 instructions, if you're using a different system, you may want to [build](BUILDING.md) nGraph from sources.

Installing nGraph Python API from PyPI is simple:

    pip install ngraph-core


## Usage example

Using nGraph's Python API to construct a computation graph and execute a computation is simple. The following example shows how to create a simple `(A + B) * C` computation graph and calculate a result using 3 numpy arrays as input.

```python
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
```


[frontend_onnx]: https://pypi.org/project/ngraph-onnx/
[frontend_mxnet]: https://pypi.org/project/ngraph-mxnet/ 
[frontend_tf]: https://pypi.org/project/ngraph-tensorflow-bridge/
[ngraph_github]: github.com/NervanaSystems/ngraph "nGraph on GitHub"
[api_python]: https://ngraph.nervanasys.com/docs/latest/python_api/ "nGraph's Python API documentation"
[api_cpp]: https://ngraph.nervanasys.com/docs/latest/howto/ 
