.. introduction.rst:

.. _introduction:

Introduction
############

Future developments in :abbr:`Artificial Intelligence (AI)` will increasingly 
rely on better methods to accelerate the performance of deep learning workloads. 
As :abbr:`Deep Learning (DL)` models become more complex, and as the volume of 
data those models are expected to handle increases rapidly, the deployment of 
scalable AI solutions becomes a greater challenge. 

Today, two standard approaches to accelerate deep learning performance are:

#. **Design hardware solutions dedicated to deep learning computation** -- Many 
   companies, ranging from startups to established manufacturers such as
   Intel, are actively developing :abbr:`Application Specific Integrated Circuits (ASICs)`
   to accelerate the performance of deep learning for both training and 
   inference.

#. **Optimize software to accelerate performance** -- nGraph Compiler, an 
   open-source deep learning compiler, is Intel's solution to deliver performance 
   via software optimization. nGraph provides developers with a way to 
   accelerate workloads via software and to provide a significant increase 
   in performance for standard hardware targets such as CPUs and GPUs. For 
   deploying scalable AI solutions, nGraph uses kernel libraries, a popular 
   and effective method to improve deep learning performance. Where kernel 
   libraries are available and perform well, we use them.

Motivations 
===========

The current :abbr:`State-of-the-Art (SoTA)` software solution for deep 
learning computation is to integrate kernel libraries such as Intel® 
:abbr:`Math Kernel Library for Deep Neural Networks (Intel® MKL DNN)` 
and Nvidia's CuDNN into deep learning frameworks. These kernel 
libraries offer a performance boost during runtime on specific hardware 
targets through highly-optimized kernels and other operator-level 
optimizations.

However, kernel libraries have three main problems:

#. Kernel libraries do not support graph-level optimizations. 
#. Framework integration of kernel libraries does not scale. 
#. The number of required kernels keeps growing.

nGraph Compiler addresses the first two problems, and nGraph Compiler combined
with PlaidML addresses the third problem. nGraph applies graph-level
optimizations by taking the computational graph from a deep learning framework
such as TensorFlow and reconstructing it with nGraph's 
:abbr: `IR (Intermediate Representation)`. nGraph IR centralizes computational 
graphs from various frameworks and provides a unified way to connect backends 
for targeted hardware. To address the third problem, nGraph is integrated with 
PlaidML, a tensor compiler, which generates code in LLVM, OpenCL, OpenGL, and 
Metal. Low-level optimizations are automatically applied to the generated code, 
resulting in a more efficient execution that does not require manual kernel 
integration for most hardware targets.

The following three sections explore the main problems of kernel libraries in
more detail and describe how nGraph addresses them.

Problem 1: Kernel libraries do not support graph-level optimizations
--------------------------------------------------------------------

The example diagrams below show how a deep learning framework, when integrated
with a kernel library, can optimally run each operation in a computational
graph, but the choice of operations in the graph may not be optimal.

.. _figure-A:

.. figure:: graphics/kernel-problem-1.png
   :width: 100%
   :alt:

   **Figure A**: The mathematical operations in a Deep Learning stack can be 
   simplified significantly with a graph compiler


The computation is constructed to execute ``(A+B)*C``. With nGraph, we can
further optimize the graph to be represented as ``A*C``. From the first graph
shown on the left, the operation on the constant ``B`` can be computed at
compile time (an optimization known as *constant folding*). The graph can be
further simplified to the one on the right because the constant has a value of
zero (known as *algebraic simplification*). Without such graph-level
optimizations, a deep learning framework with a kernel library will compute
all operations, resulting in suboptimal execution.

Problem 2: Framework integration of kernel libraries does not scale
-------------------------------------------------------------------

Due to the growing number of new deep learning accelerators, integrating
kernel libraries with frameworks has become increasingly more difficult. For
each new deep learning accelerator, a custom kernel library integration must
be implemented by a team of experts. This labor-intensive work is further
complicated by the number of frameworks, as illustrated in the following
diagram.

.. _figure-B:

.. figure:: graphics/kernel-problem-2.png
   :width: 100%
   :alt:

   **Figure B**:  A many-to-many problem

Each framework must be manually integrated with each hardware-specific kernel
library. Additionally, each integration is unique to the framework and its set
of deep learning operators, view on memory layout, feature set, etc. Each
connection that needs to be made increases the amount of work, resulting in a
fragile setup that is costly to maintain.

nGraph solves this problem with bridges. A bridge takes a computational
graph or similar structure and reconstructs it in the nGraph IR along with a
few primitive nGraph operations. With a unified computational graph, kernel
libraries no longer need to be separately integrated into each deep learning
framework. Instead, the libraries only need to support nGraph primitive
operations, and this approach streamlines the integration process for the
backend.


Problem 3: The number of required kernels keeps growing
-------------------------------------------------------

Integrating kernel libraries with multiple deep learning frameworks is a
difficult task that becomes more complex with the growing number of
kernels needed to achieve optimal performance. Past deep learning research has
been built on a small set of standard computational primitives (convolution,
GEMM, etc.). But as AI research advances and industrial deep learning
applications continue to develop, the number of required kernels continues to
increase exponentially. The number of required kernels is based on the number
of chip designs, data types, operations, and the cardinality of each parameter
per operation. Each connection in the following diagram represents significant 
work for what will ultimately be a fragile setup that is costly to maintain.

.. _figure-C:

.. figure:: graphics/kernel-problem-3.png
   :width: 100%
   :alt:

   **Figure C**:  Inevitable scaling problem



Solution: A customizable graph compiler for complex operations
==============================================================

`OpenVINO toolkit`_ is powered by nGraph capabilities for Graph compilation. 
To represent a :abbr:`DL (Deep Learning)` model in real-time and perform 
complex operations on that model, users can `build an nGraph function`_.

Once created, it can wrap into a ``CNNNetwork``, creating utility for data 
scientists or application developers to use operations that do not depend 
on existing Deep Learning (DL) frameworks. 
 

.. _OpenVINO toolkit: http://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html
.. _build an nGraph function: http://docs.openvinotoolkit.org/latest/_docs_IE_DG_nGraphTutorial.html
.. _add custom operations: http://docs.openvinotoolkit.org/latest/_docs_IE_DG_AddingNGraphOps.html