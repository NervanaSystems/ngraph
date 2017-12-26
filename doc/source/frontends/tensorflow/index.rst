.. /frontends/tensorflow/index.rst

Tensorflow 
==========

frontend integration notes
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
========

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