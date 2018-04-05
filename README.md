# nGraph library [![Build Status][build-status-badge]][build-status]

Welcome to Intel® nGraph™, an open source C++ library, compiler and
runtime. This project enables modern compute platforms to run and
train Deep Neural Network (DNN) models. It is framework-neutral and
supports a variety of backends used by Deep Learning (DL) frameworks.

![nGraph ecosystem][ngraph-ecosystem]


|Framework   | bridge available? | ONNX support?  |
|------------|-------------------|----------------|
| neon       | yes               | yes            |
| MXNet*     | yes               | yes            |
| TensorFlow*| yes               | yes            |
| PyTorch*   | not yet           | yes            |
| CNTK*      | not yet           | yes            |
| Caffe2*    | not yet           | yes            |


## Documentation

See our [install] docs for how to get started.

For this early release, we provide [framework integration guides] to
compile MXNet and TensorFlow-based projects. If you already have a
trained model, we've put together a getting started guide for
[how to import] a deep learning model and start working with the nGraph
APIs.

## Support

Please submit your questions, feature requests and bug reports via
[GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea how
to improve the library:

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

[install]: http://ngraph.nervanasys.com/docs/latest/install.html
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: http://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph-ecosystem]: doc/sphinx/source/graphics/ngraph-ecosystem.png "nGraph Ecosystem"
[build-status]: https://travis-ci.org/NervanaSystems/ngraph/branches
[build-status-badge]: https://travis-ci.org/NervanaSystems/ngraph.svg?branch=master
