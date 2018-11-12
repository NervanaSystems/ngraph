# nGraph Compiler Stack 
## Version: Beta (1.0) 

[![Build Status][build-status-badge]][build-status] [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/ngraph/blob/master/LICENSE)
 

![nGraph Compiler][ngraph-compiler-stack-readme.png]


<div align="center">
  <h6>
    <a href="https://ngraph.nervanasys.com/docs/latest/project/about.html">
      Architecture and Features</a><span> | </span>
    <a href="https://ngraph.nervanasys.com/docs/latest/project/release-notes.html">
      Release Notes</a><span> | </span>
    <a href="https://ngraph.nervanasys.com/docs/latest">Documentation</a><span> | </span>
    <a href="#Ecosystem" >nGraph Ecosystem</a><span> | </span>
    <a href="#Getting-started-guides" >Getting Started Guides</a><span> | </span>
    <a href="#How-to-contribute" >How to Contribute</a>
 </h6>
</div>


## Getting started guides


|  Framework / Version       | Installation guide                     | Notes  
|----------------------------|----------------------------------------|-----------------------------------
| TensorFlow* 1.12           | [Pip package] or [Build from source]   | 17 [Validated workloads]
| MXNet* 1.4                 | [Enable the module] or [Source compile]| 17 [Validated workloads] 
| PaddlePaddle* Fluid        | Coming soon                            | To be determined  
| ONNX 1.3                   | [Pip package] or [Build from source]   | 13 [Functional] workloads with DenseNet-121, Inception-v1, ResNet-50, Inception-v2, ShuffleNet, SqueezeNet, VGG-19, and 7 more   

Frameworks using nGraph Compiler stack to execute workloads have shown **3X** to 
**45X** performance boost when compared to native framework implementations. 
We've also seen performance boosts running workloads that are not included on 
the list of [Validated workloads], thanks to our powerful subgraph pattern 
matching and thanks to the collaborative efforts we've put into the DL community, 
such as with [nGraph-ONNX adaptable] APIs and [nGraph for PyTorch developers].

Additional work is also being done via [PlaidML] which will feature running compute 
for Deep Learning with GPU accleration and support for MacOS. See our [Architecture and features]
for what the stack looks like today and watch our [Release Notes] for recent 
changes.


| Backend                                       | current support   | future support |
|-----------------------------------------------|-------------------|----------------|
| Intel® Architecture CPU                       | yes               | yes            |
| Intel® Nervana™ Neural Network Processor (NNP)| yes               | yes            |
| Intel [Movidius™ Myriad™ 2] VPUs              | coming soon       | yes            |
| Intel® Architecture GPUs                      | via PlaidML       | yes            |
| AMD* GPUs                                     | via PlaidML       | yes            |
| NVIDIA* GPUs                                  | via PlaidML       | some           | 
| Field Programmable Gate Arrays (FPGA)         | no                | yes            |


## Ecosystem

![nGraph ecosystem][ngraph-ecosystem]

The **nGraph Compiler** is Intel's graph compiler for Artificial Neural Networks. 
Documentation in this repo describes how you can program any framework 
to run training and inference computations on a variety of Backends including 
Intel® Architecture Processors (CPUs), Intel® Nervana™ Neural Network Processors 
(NNPs), cuDNN-compatible graphics cards (GPUs), custom VPUs like Movidius, and
many others. The default CPU Backend also provides an interactive *Interpreter* 
mode that can be used to zero in on a DL model and create custom nGraph 
optimizations that can be used to further accelerate training or inference, in 
whatever scenario you need. nGraph provides both a C++ API for framework 
developers and a Python API which can run inference on models imported from 
ONNX. 


## Documentation

See our [build the Library] docs for how to get started.

For this early release, we provide [framework integration guides] to
compile MXNet and TensorFlow-based projects. If you already have a
trained model, we've put together a getting started guide for
[how to import] a deep learning model and start working with the nGraph
APIs.

## Support

Please submit your questions, feature requests and bug reports via
[GitHub issues].

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
[ngraph-compiler-stack]: doc/sphinx/source/graphics/ngraph-compiler-stack.png "nGraph Compiler Stack"
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

