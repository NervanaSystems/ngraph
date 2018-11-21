# nGraph Compiler Stack Beta

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/ngraph/blob/master/LICENSE) [![Build Status][build-status-badge]][build-status] 

<div align="left">
  <h3>
    <a href="https://ngraph.nervanasys.com/docs/latest/project/about.html">
      Architecture and features</a> | <a href="#Ecosystem" >nGraph ecosystem</a><span> </span> <span> | </span> 
    <a href="https://ngraph.nervanasys.com/docs/latest/project/release-notes.html">
      Beta release notes</a><span> | </span> <br />
    <a href="https://ngraph.nervanasys.com/docs/latest">Documentation</a><span> | </span>
    <a href="#How-to-contribute" >How to contribute</a>
 </h3>
</div>

## Quick start


To begin using nGraph with popular frameworks to accelerate deep learning 
workloads on CPU for inference, please refer to the links below. 

|  Framework / Version       | Installation guide                     | Notes  
|----------------------------|----------------------------------------|-----------------------------------
| TensorFlow* 1.12           | [Pip package] or [Build from source]   | 17 [Validated workloads]
| MXNet* 1.4                 | [Enable the module] or [Source compile]| 17 [Validated workloads]   
| ONNX 1.3                   | [Pip package]                          | 13 [Functional] workloads with DenseNet-121, Inception-v1, ResNet-50, Inception-v2, ShuffleNet, SqueezeNet, VGG-19, and 7 more   

Frameworks using nGraph Compiler stack to execute workloads have shown 
**3X** to **45X** performance boost when compared to native framework 
implementations. We've also seen performance boosts running workloads that 
are not included on the list of [Validated workloads], thanks to our 
powerful subgraph pattern matching and thanks to the collaborative efforts 
we've put into the DL community, such as with [nGraph-ONNX adaptable] APIs 
and [nGraph for PyTorch developers].

Additional work is also being done via [PlaidML] which will feature running 
compute for Deep Learning with GPU accleration and support for MacOS. See our 
[Architecture and features] for what the stack looks like today and watch our 
[Release Notes] for recent changes.


## What is nGraph Compiler? 

nGraph Compiler aims to accelerate developing and deploying AI workloads 
using any deep learning framework with a variety of hardware targets. 
We strongly believe in providing freedom, performance, and ease-of-use to AI 
developers. 

The diagram below shows what deep learning frameworks and hardware targets
we support. More details on these current and future plans are in the ecosystem
section. 


![nGraph ecosystem][ngraph-ecosystem]


While the ecosystem shown above is all functioning, we have validated 
performance metrics for deep learning inference on CPU processors including 
as Intel® Xeon®. Please refer to the [Beta release notes] to learn more. 
The Gold release is targeted for April 2019; it will feature broader workload 
coverage, including support for quantized graphs, and more detail on our 
advanced support for ``int8``. 

Our documentation has extensive information about how to use nGraph Compiler 
stack to create an nGraph computational graph, integrate custom frameworks, 
and interact with supported backends. If you wish to contribute to the 
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

![nGraph Compiler Stack][ngraph-compiler-stack-readme]

| Backend                                       | current support   | future support |
|-----------------------------------------------|-------------------|----------------|
| Intel® Architecture CPU                       | yes               | yes            |
| Intel® Nervana™ Neural Network Processor (NNP)| yes               | yes            |
| Intel [Movidius™ Myriad™ 2] VPUs              | coming soon       | yes            |
| Intel® Architecture GPUs                      | via PlaidML       | yes            |
| AMD* GPUs                                     | via PlaidML       | yes            |
| NVIDIA* GPUs                                  | via PlaidML       | some           | 
| Field Programmable Gate Arrays (FPGA)         | no                | yes            |



[Architecture and features]:https://ngraph.nervanasys.com/docs/latest/project/about.html
[Documentation]: https://ngraph.nervanasys.com/docs/latest
[build the Library]: https://ngraph.nervanasys.com/docs/latest/buildlb.html
[Getting Started Guides]: Getting-started-guides
[Validated workloads]: https://ngraph.nervanasys.com/docs/latest/frameworks/validation-testing.html
[Functional]: https://github.com/NervanaSystems/ngraph-onnx/ 
[How to contribute]: How-to-contribute
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[release notes]: https://ngraph.nervanasys.com/docs/latest/project/release-notes.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[contrib guide]: https://ngraph.nervanasys.com/docs/latest/project/code-contributor-README.html
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: https://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph-ecosystem]: doc/sphinx/source/graphics/599px-Intel-ngraph-ecosystem.png "nGraph Ecosystem"
[ngraph-compiler-stack-readme]: doc/sphinx/source/graphics/ngraph-compiler-stack-readme.png "nGraph Compiler Stack"
[build-status]: https://travis-ci.org/NervanaSystems/ngraph/branches
[build-status-badge]: https://travis-ci.org/NervanaSystems/ngraph.svg?branch=master
[develop-without-lockin]: doc/sphinx/source/graphics/develop-without-lockin.png "Develop on any part of the stack wtihout lockin"
[Movidius™ Myriad™ 2]:https://www.movidius.com/solutions/vision-processing-unit
[PlaidML]: https://github.com/plaidml/plaidml
[Pip package]: https://github.com/NervanaSystems/ngraph-onnx#installing-ngraph-onnx
[Build from source]: https://github.com/NervanaSystems/ngraph-tf
[Source compile]: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/NGRAPH_README.md
[nGraph-ONNX]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/README.md
[nGraph-ONNX adaptable]: https://ai.intel.com/adaptable-deep-learning-solutions-with-ngraph-compiler-and-onnx/
[nGraph for PyTorch developers]: https://ai.intel.com/investing-in-the-pytorch-developer-community
[Validated workloads]: https://ngraph.nervanasys.com/docs/latest/frameworks/validation-testing.html

