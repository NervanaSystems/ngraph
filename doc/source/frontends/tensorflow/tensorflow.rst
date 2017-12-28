.. tensorflow:

.. ---------------------------------------------------------------------------
.. Copyright 2018 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

TensorFlow\*
============

Intel® nGraph™ library permits interoperability with `the TensorFlow frontend`_ 
framework. The TensorFlow\* importer allows users to define models in TensorFlow 
and to then execute computations using transformers from the Intel nGraph™ backend.  

TensorFlow\* can also be built directly on Intel CPUs; for tips on this see our 
documentation for :doc:`tensorflow-opts-ia`.

Minimal example
---------------

Here's a minimal example for the TensorFlow importer

.. code-block:: python

   from __future__ import print_function
   from ngraph.frontends.TensorFlow.tf_importer.importer import TFImporter
   import ngraph.transformers as ngt
   import TensorFlow as tf
   import ngraph as ng

   # TensorFlow ops
   x = tf.constant(1.)
   y = tf.constant(2.)
   f = x + y

   # import
   importer = TFImporter()
   importer.import_graph_def(tf.Session().graph_def)

   # get handle
   f_ng = importer.get_op_handle(f)

   # execute
   transformer = ngt.make_transformer()
   f_result = transformer.computation(f_ng)()
   print(f_result)



Walkthrough of MNIST MLP example
--------------------------------

Here's a walkthrough of the MNIST MLP example. For the full source code of the
example, refer to the
`examples <https://github.com/NervanaSystems/ngraph/tree/master/ngraph/frontends/tensorflow/examples/>`__
directory.

1. Define the MNIST MLP model in TensorFlow as shown below::

    x = tf.placeholder(tf.float32, [args.batch_size, 784])
    t = tf.placeholder(tf.float32, [args.batch_size, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b
    cost = tf.reduce_mean(-tf.reduce_sum(
        t * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    init = tf.initialize_all_variables()

In this example, we need to explicitly set ``init`` to
``tf.initialize_all_variables()`` since we need to use the handle of the
``init`` op for the nGraph library to execute the correct initialization.

2. Import TensorFlow ``GraphDef``::

    importer = TFImporter()
    importer.import_graph_def(tf.Session().graph_def)

- We use the ``TFImporter.import_graph_def()`` function to import from
  TensorFlow sessions's ``graph_def``.
- The importer also supports importing from a ``graph_def`` protobuf file
  using ``TFImporter.import_protobuf()``. For example, a ``graph_def`` file can
  be dumped by ``tf.train.SummaryWriter()``.

3. Get the handles of the corresponding the nGraph library ops::

    x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle([x, t, cost, init])

TensorFlow nodes are converted to the nGraph library ops. To evaluate a
TensorFlow node, we need to get its corresponding the nGraph library node using
``TFImporter.get_op_handle()``.

4. Perform autodiff and define computations::

    updates = SGDOptimizer(args.lrate).minimize(cost_ng)
    transformer = ngt.make_transformer()
    train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
    init_comp = transformer.computation(init_op_ng)
    transformer.initialize()

As we only import the forward graph from TensorFlow, we should use the nGraph library's
autodiff to compute gradients and get optimizers.

5. Training using the nGraph library::

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    init_comp()
    for idx in range(args.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
        cost_val, _ = train_comp(batch_xs, batch_ys)
        print("[Iter %s] Cost = %s" % (idx, cost_val))

Now we can train the model in ngraph as if it were a native the nGraph library model. All
the nGraph library functionalities and syntax can be applied after the graph is imported.

6. Training using TensorFlow as comparison::

    with tf.Session() as sess:
        # train in tensorflow
        train_step = tf.train.GradientDescentOptimizer(args.lrate).minimize(cost)
        sess.run(init)

        mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val, _ = sess.run([cost, train_step],
                                   feed_dict={x: batch_xs, t: batch_ys})
            print("[Iter %s] Cost = %s" % (idx, cost_val))

Finally, we train the model using standard TensorFlow. The the nGraph library results above
match TensorFlow's results.


Current limitations
-------------------

1. Only a subset of operations are supported.

  - Currently we only support a subset of operations from TensorFlow that are
    related to neural networks. We are working on getting more ops supported in
    the importer.
  - A util function ``TFImporter._get_unimplemented_ops()`` is provided for
    getting a list of unimplemented ops from a particular model.

2. The importer should be used to import the forward graph.

  - User should use the importer to import the forward pass of the TensorFlow graph,
    and then perform autodiff and training updates in the nGraph library.
  - TensorFlow ops related to gradient computation are not supported.
  - In the future, bidirectional weight exchange between TensorFlow and the nGraph library will
    also be supported.

3. Staticness

  - In the nGraph library, the transformer can alter the computation graph during the
    transformation phase, thus we need to declare all computations before
    executing any of them. Altering the imported graph after transformer
    initialization is not supported.
  - TensorFlow allows dynamic parameters to its ops. For example, the kernel
    size of a ``Conv2d`` can be the result of another computation. Since
    the nGraph library needs to know dimension information prior to execution to allocating
    memory, dynamic parameters are not supported in importer.


Frontend integration notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow\* is a leading deep learning and machine learning framework,
which makes it important for Intel and Google to ensure that it is able
to extract maximum performance from Intel’s hardware offering.
Here we introduce the Artificial Intelligence (AI) community to
TensorFlow optimizations on Intel® Xeon® and Intel® Xeon Phi™
processor-based platforms. These optimizations are the fruit of a close
collaboration between Intel and Google engineers announced in 2016 by
Intel’s Diane Bryant and Google’s Diane Green at the first Intel AI Day.

This section describes some performance challenges that we encountered
during this optimization exercise and the solutions adopted. We also
report out performance improvements on a sample of common neural
networks models. These optimizations can result in orders of magnitude
higher performance. For example, our measurements are showing up to 70x
higher performance for training and up to 85x higher performance for
inference on Intel® Xeon Phi™ processor 7250 (KNL). Intel® Xeon®
processor E5 v4 (BDW) and Intel Xeon Phi processor 7250-based platforms,
they lay the foundation for next generation products from Intel. In
particular, users are expected to see improved performance on Intel Xeon
(code named Skylake) and Intel Xeon Phi (code named Knights Mill) coming
out later this year.

Optimizing deep learning models performance on modern CPUs presents a
number of challenges not very different from those seen when optimizing
other performance-sensitive applications in High Performance Computing
(HPC):

#. Code refactoring needed to take advantage of modern vector
   instructions. This means ensuring that all the key primitives, such
   as convolution, matrix multiplication, and batch normalization are
   vectorized to the latest SIMD instructions (AVX2 for Intel Xeon
   processors and AVX512 for Intel Xeon Phi processors).
#. Maximum performance requires paying special attention to using all
   the available cores efficiently. Again this means looking at
   parallelization within a given layer or operation as well as
   parallelization across layers.
#. As much as possible, data has to be available when the execution
   units need it. This means balanced use of prefetching, cache blocking
   techniques and data formats that promote spatial and temporal
   locality.

To meet these requirements, Intel developed a number of optimized deep
learning primitives that can be used inside the different deep learning
frameworks to ensure that we implement common building blocks
efficiently. In addition to matrix multiplication and convolution, these
building blocks include:

-  Direct batched convolution
-  Inner product
-  Pooling: maximum, minimum, average
-  Normalization: local response normalization across channels (LRN),
   batch normalization
-  Activation: rectified linear unit (ReLU)
-  Data manipulation: multi-dimensional transposition (conversion),
   split, concat, sum and scale.

Details on `DNN Primitives on Intel MKL`_ optimized primitives.

In TensorFlow, we implemented Intel-optimized versions of operations to
make sure that these operations can leverage Intel MKL-DNN primitives
wherever possible. While, this is a necessary step to enable scalable
performance on Intel® architecture, we also had to implement a number of
other optimizations. In particular, Intel MKL uses a different layout
than the default layout in TensorFlow for performance reasons. We needed
to ensure that the overhead of conversion between the two formats is
kept to a minimum. We also wanted to ensure that data scientists and
other TensorFlow users don’t have to change their existing neural
network models to take advantage of these optimizations.


Colophon
~~~~~~~~~

Special thanks to contributors of `this whitepaper`_. 
  
.. Intel contributors -- Elmoustapha Ould-Ahmed-Vall, Mahmoud Abuzaina, Md Faijul
   Amin, Jayaram Bobba, Roman S Dubtsov, Evarist M Fomenko, Mukesh
   Gangadhar, Niranjan Hasabnis, Jing Huang, Deepthi Karkada, Young Jin
   Kim, Srihari Makineni, Dmitri Mishura, Karthik Raman, AG Ramesh, Vivek
   V Rane, Michael Riera, Dmitry Sergeev, Vamsi Sripathi, Bhavani
   Subramanian, Lakshay Tokas, Antonio C Valles
 
.. Google contributors -- Andy Davis, Toby Boyd, Megan Kacholia, Rasmus Larsen,
   Rajat Monga, Thiru Palanisamy, Vijay Vasudevan, Yao Zhang



.. _this whitepaper: https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture
.. _DNN Primitives on Intel MKL: https://software.intel.com/en-us/articles/introducing-dnn-primitives-in-intelr-mkl
.. _the Tensorflow frontend: https://www.tensorflow.org
