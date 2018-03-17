# Intel® nGraph™ library 

Welcome to Intel® nGraph™, an open source C++ library and compiler. This 
project enables modern compute platforms to run and train Deep Neural Network 
(DNN) models. It is framework-neutral and supports a variety of backends 
used by Deep Learning (DL) frameworks. 

The nGraph library translates a framework’s representation of computations into 
an Intermediate Representation (IR) designed to promote computational efficiency 
on target hardware. Initially-supported backends include Intel Architecture CPUs, 
the Intel® Nervana Neural Network Processor™ (NNP), and NVIDIA\* GPUs. 
Currently-supported compiler optimizations include efficient memory management 
and data layout abstraction. 

## Documentation

See our [install] docs for how to get started. 

For this early release, we provide [framework integration guides] to compile 
MXNet and TensorFlow-based projects.  

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea how to improve the library:

* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch
* In the case of a larger feature, create a test
* Submit a [pull request]
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged the repository.

[install]: http://ngraph.nervanasys.com/docs/latest/install.html
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
