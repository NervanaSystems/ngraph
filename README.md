# nGraph Compiler Stack (Beta)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/ngraph/blob/master/LICENSE) [![Build Status][build-status-badge]][build-status] 

<div align="left">
  <h4>
    <a href="./ABOUT.md">Architecture & features</a> | <a href="./ecosystem-overview.md" >Ecosystem</a> | <a href="https://ngraph.nervanasys.com/docs/latest/project/release-notes.html">Release notes</a><span> | </span> <a href="https://ngraph.nervanasys.com/docs/latest">Documentation</a><span> | </span> <a href="#How-to-contribute" >Contribution guide</a>
 </h4>
</div>

## Quick start


To begin using nGraph with popular frameworks to accelerate deep learning 
workloads on CPU for inference, please refer to the links below. 

|  Framework (Version)       | Installation guide                     | Notes  
|----------------------------|----------------------------------------|-----------------------------------
| TensorFlow* 1.12           | [Pip install](https://github.com/NervanaSystems/ngraph-tf/tree/v0.8.0#option-1-use-a-pre-built-ngraph-tensorflow-bridge) or [Build from source](https://github.com/NervanaSystems/ngraph-tf/tree/v0.8.0#option-2-build-ngraph-bridge-from-source-using-tensorflow-source)   | 20 [Validated workloads]
| MXNet* 1.3                 | [Pip install](https://github.com/NervanaSystems/ngraph-mxnet#Installation) or [Build from source](https://github.com/NervanaSystems/ngraph-mxnet#building-with-ngraph-support)| 18 [Validated workloads]   
| ONNX 1.3                   | [Pip install](https://github.com/NervanaSystems/ngraph-onnx#installation)                          | 14 [Validated workloads] 

:exclamation: :exclamation: :exclamation: Note that the ``pip`` package option 
works only with Ubuntu 16.04 or greater and Intel® Xeon® CPUs. CPUs without 
Intel® Advanced Vector Extensions 512 (Intel® AVX-512) will not run these 
packages; the alternative is to build from source. Wider support for other 
CPUs will be offered starting in early 2019 :exclamation: :exclamation: :exclamation:

Frameworks using nGraph Compiler stack to execute workloads have shown 
[**up to 45X**](https://ai.intel.com/ngraph-compiler-stack-beta-release/) 
performance boost when compared to native framework implementations. We've also 
seen performance boosts running workloads that are not included on the list of 
[Validated workloads], thanks to our powerful subgraph pattern matching.

Additional work is also being done via [PlaidML] which will feature running 
compute for Deep Learning with GPU accleration. See our 
[Architecture and features] for what the stack looks like today and watch our 
[Release Notes] for recent changes.


## What is nGraph Compiler? 

nGraph Compiler aims to accelerate developing and deploying AI workloads 
using any deep learning framework with a variety of hardware targets. 
We strongly believe in providing freedom, performance, and ease-of-use to AI 
developers. 

The diagram below shows what deep learning frameworks and hardware targets
we support. More details on these current and future plans are in the [ecosystem]
section. 


![nGraph wireframe][ngraph_wireframes_with_notice]


While the ecosystem shown above is all functioning, we have validated 
performance for deep learning inference on CPU processors such as Intel® Xeon®. 
Please refer to the [Release notes] to learn more. The Gold release 
is targeted for April 2019; it will feature broader workload coverage, 
including quantized graphs, and more detail on our advanced support for 
``int8``. 

Our documentation has extensive information about how to use nGraph Compiler 
stack to create an nGraph computational graph, integrate custom frameworks, 
and to interact with supported backends. If you wish to contribute to the 
project, please don't hesitate to ask questions in [GitHub issues] after 
reviewing our contribution guide below. 


## How to contribute

We welcome community contributions to nGraph. If you have an idea how
to improve it:

* See the [contrib guide] for code formatting and style guidelines.
* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* Make sure your PR passes all CI tests. Note: our [Travis-CI][build-status] service
  runs only on a CPU backend on Linux. We will run additional tests
  in other environments.
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.


[Ecosystem]: ./ecosystem-overview.md
[Architecture and features]:https://ngraph.nervanasys.com/docs/latest/project/about.html
[Documentation]: https://ngraph.nervanasys.com/docs/latest
[build the Library]: https://ngraph.nervanasys.com/docs/latest/buildlb.html
[Getting Started Guides]: Getting-started-guides
[Validated workloads]: https://ngraph.nervanasys.com/docs/latest/frameworks/validation.html
[Functional]: https://github.com/NervanaSystems/ngraph-onnx/ 
[How to contribute]: How-to-contribute
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[release notes]: https://ngraph.nervanasys.com/docs/latest/project/release-notes.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[contrib guide]: https://ngraph.nervanasys.com/docs/latest/project/contribution-guide.html
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: https://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph_wireframes_with_notice]: doc/sphinx/source/graphics/readme_stack.png "nGraph wireframe"
[ngraph-compiler-stack-readme]: doc/sphinx/source/graphics/ngraph-compiler-stack-readme.png "nGraph Compiler Stack"
[build-status]: https://travis-ci.org/NervanaSystems/ngraph/branches
[build-status-badge]: https://travis-ci.org/NervanaSystems/ngraph.svg?branch=master
[develop-without-lockin]: doc/sphinx/source/graphics/develop-without-lockin.png "Develop on any part of the stack wtihout lockin"
[Movidius™ Myriad™ 2]:https://www.movidius.com/solutions/vision-processing-unit
[PlaidML]: https://github.com/plaidml/plaidml
[Source compile]: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/README.md
[nGraph-ONNX]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/README.md
[nGraph-ONNX adaptable]: https://ai.intel.com/adaptable-deep-learning-solutions-with-ngraph-compiler-and-onnx/
[nGraph for PyTorch developers]: https://ai.intel.com/investing-in-the-pytorch-developer-community
[Validated workloads]: https://ngraph.nervanasys.com/docs/latest/frameworks/genre-validation.html


