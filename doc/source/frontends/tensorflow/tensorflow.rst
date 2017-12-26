.. _tensorflow:

.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
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

Intel® nGraph™ abstraction layer permits interoperability with 
`the TensorFlow frontend`_ framework. The TensorFlow\* importer allows users to 
define a limited set of models in TensorFlow and to then execute computations 
using transformers from the Intel nGraph™ backend.


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
``init`` op for Intel Nervana graph to execute the correct initialization.

2. Import TensorFlow ``GraphDef``::

    importer = TFImporter()
    importer.import_graph_def(tf.Session().graph_def)

- We use the ``TFImporter.import_graph_def()`` function to import from
  TensorFlow sessions's ``graph_def``.
- The importer also supports importing from a ``graph_def`` protobuf file
  using ``TFImporter.import_protobuf()``. For example, a ``graph_def`` file can
  be dumped by ``tf.train.SummaryWriter()``.

3. Get the handles of the corresponding Intel Nervana graph ops::

    x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle([x, t, cost, init])

TensorFlow nodes are converted to Intel Nervana graph ops. To evaluate a
TensorFlow node, we need to get its corresponding Intel Nervana graph node using
``TFImporter.get_op_handle()``.

4. Perform autodiff and define computations::

    updates = SGDOptimizer(args.lrate).minimize(cost_ng)
    transformer = ngt.make_transformer()
    train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
    init_comp = transformer.computation(init_op_ng)
    transformer.initialize()

As we only import the forward graph from TensorFlow, we should use Intel Nervana graph's
autodiff to compute gradients and get optimizers.

5. Training using Intel Nervana graph::

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    init_comp()
    for idx in range(args.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
        cost_val, _ = train_comp(batch_xs, batch_ys)
        print("[Iter %s] Cost = %s" % (idx, cost_val))

Now we can train the model in ngraph as if it were a native Intel Nervana graph model. All
Intel Nervana graph functionalities and syntax can be applied after the graph is imported.

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

Finally, we train the model using standard TensorFlow. The Intel Nervana graph results above
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
    and then perform autodiff and training updates in Intel Nervana graph.
  - TensorFlow ops related to gradient computation are not supported.
  - In the future, bidirectional weight exchange between TensorFlow and Intel Nervana graph will
    also be supported.

3. Staticness

  - In Intel Nervana graph, the transformer can alter the computation graph during the
    transformation phase, thus we need to declare all computations before
    executing any of them. Altering the imported graph after transformer
    initialization is not supported.
  - TensorFlow allows dynamic parameters to its ops. For example, the kernel
    size of a ``Conv2d`` can be the result of another computation. Since
    Intel Nervana graph needs to know dimension information prior to execution to allocating
    memory, dynamic parameters are not supported in importer.


.. _the Tensorflow frontend: https://www.tensorflow.org
