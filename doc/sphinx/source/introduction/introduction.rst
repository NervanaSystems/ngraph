.. introduction:

Introduction
############

For making further developments in Artificial Intelligence (AI), developers
need better methods to accelerate the performance of deep learning. Deep
learning models are becoming more complex and the sizes of their data sets are
rapidly increasing, making the deployment of scalable AI solutions a greater
challenge.

Currently, there are two standard approaches for developers to accelerate deep
learning performance:

**Design hardware solutions dedicated to deep learning computation**

Established chip manufacturers such as Intel and hardware startups are
actively developing Application Specific Integrate Circuits (ASICs) to
accelerate the performance of deep learning for both training and inference.

**Optimize software to accelerate performance**

nGraph, an open-source graph compiler, is our solution to deliver performance
via software optimization. nGraph was created to provide developers with a way
to accelerate the performance of upcoming, deep learning ASICs via software
and to provide a significant increase in performance for standard hardware
targets such as CPUs and GPUs. For deploying scalable AI solutions, nGraph
uses kernel libraries which are currently a popular method to improve deep
learning performance. 

Motivations
===========

The current state-of-the-art software solution for speeding up deep learning
computation is to integrate kernel libraries such as Intel® Math Kernel
Library for Deep Neural Networks (Intel® MKL DNN) and Nvidia\*'s CuDNN into
deep learning frameworks. These kernel libraries offer a performance boost
during runtime on specific hardware targets through highly-optimized kernels
and other operator-level optimizations.

However, kernel libraries have three main problems: 

#. Kernel libraries do not support graph-level optimizations.
#. Framework integration of kernel libraries does not scale.
#. There are too many kernels to write, and they require expert knowledge.

The nGraph Compiler stack is designed to address the first two problems.
nGraph applies graph-level optimizations by taking the computational graph
from a deep learning framework such as TensorFlow\* and reconstructs it with
nGraph :abbr:`IR (Intermediate Representation)`. nGraph IR centralizes
computational graphs from various frameworks and provides a unified way to
connect backends for targeted hardware. To address the third problem, nGraph
is integrated with PlaidML, *a lorem ipsum*, to generate code in LLVM, OpenCL,
OpenGL, Cuda, and Metal. Low-level optimizations are automatically applied to
the generated code, resulting in a more efficient execution that does not
require manual kernel integration for most hardware targets. 

The following three sections explore the main problems of kernel libraries in
more detail and describe how nGraph addresses them.

Problem 1: Absence of graph-level optimizations
-----------------------------------------------

The example diagram below shows how a deep learning framework, when integrated
with a kernel library, can optimally run each operation in a computational
graph, but the graph itself may not be optimal.

.. _figure-A:

.. figure:: ../graphics/intro_graph_optimization.png
   :width: 555px
   :alt: 

The computation is constructed to execute ``(A+B)*C``, but in the context of 
nGraph, we can further optimize the graph to be represented as ``A*C``. From
the first graph shown on the left, the operation on the constant ``B`` can be
computed at the compile time (known as constant folding), and the graph can be
further simplified to the one on the right because the constant has a value of
zero. Without such graph-level optimizations, a deep learning framework with a
kernel library will compute all operations, resulting in suboptimal execution. 

Problem 2: Reduced scalability 
------------------------------

Due to the growing number of new deep learning accelerators, integrating
kernel libraries with frameworks has become increasingly nontrivial. For each
new deep learning accelerator, a custom kernel library integration must be
implemented by a team of experts. This labor-intensive work is further
complicated by the number of frameworks as illustrated in the diagram below
using orange lines. 

.. _figure-B:

.. figure:: ../graphics/lorem-ipsum.png
   :width: 555px
   :alt: 

Each framework must be manually integrated with each hardware-specific kernel
library. Additionally, each integration is unique to the framework and its set
of deep learning operators, view on memory layout, feature set, etc. Each
connection that needs to be made increases the amount of work, resulting in a
fragile setup that is costly to maintain.

nGraph solves this problem with nGraph bridges. A bridge takes a computational
graph from a supported framework and reconstructs it in the nGraph IR with a
few primitive nGraph operations. With a unified computational graph, kernel
libraries no longer need to be separately integrated into each deep learning
framework. Instead, the ibraries only need to support nGraph primitive
operations, and this approach streamlines the integration process for the
backend.  


Problem 3: Increasing the number of kernels 
-------------------------------------------

As previously mentioned, Kernel libraries need to be integrated with multiple
deep learning frameworks, and this already arduous task becomes even harder
due to the greater number of a kernels for achieving optimal performance. The
number of required kernels is based on the number of chip designs, data types,
operations, and the cardinality of each parameter per operation. In the past,
the number of required kernels was limited, but as AI research and industry
continue to rapidly develop, the number of required kernels is exponentially
increasing. 

.. _figure-C:

.. figure:: ../graphics/intro_kernel_explosion.png
   :width: 555px
   :alt: 

   Each of these connections represents significant work for what will
   ultimately be a fragile setup that is costly to maintain.


PlaidML addresses the exponential growth of required kernels. It takes two
inputs: the operation defined by the user and the machine description of the
hardware target. 

PlaidML uses Tile, a :abbr:Domain-Specific Language (DSL) that allows
developers to express how an operation should calculate tensors in an
intuitive, mathematical form. PlaidML takes user-defined Tile code along with
the machine description (threads, max memory input, etc.) to automatically
apply low-level optimizations. These automated optimizations do not require
kernel developers to write kernel libraries, easing their burden. Integrating
PlaidML with nGraph provides flexbility to support newer deep learning models
in the absence of hand-optimized kernels for new operations.

Solution: nGraph and PlaidML
============================

We developed nGraph and integrated it with PlaidML to allow developers to
accelerate deep learning performance and address the problem of scalable
kernel libraries. 

To address the problem of scaling backends, nGraph applies graph-level
optimizations to deep learning computations and unifies computational graphs
from deep learning frameworks with nGraph IR. 

In conjuction with Ngraph's graph-level optimizations, PlaidML automatically
applies low-level optimizations to improve deep learning performance.
Additionally, PlaidML offers extensive support for various hardware targets
due to its abilility to generate code in LLVM, OpenCL, OpenGL, CUDA, and Metal.

Given a backend with existing kernel libraries, nGraph can readily support the
target hardware because the backend only needs to support a few primitive
operations. If the hardware supports one of the coding languages supported by
PlaidML, developers need to specify the machine description to support the
hardware. Together, nGraph and PlaidML provide the best of both worlds.

This documentation provides technical details of nGraph's core functionality
as well as framework and backend integrations. Creating a compiler stack like
nGraph and PlaidML requires expert knowledge, and we're confident that nGraph
and PlaidML will make life easier for many kinds of developers: 

#. Framework owners looking to support new hardware and custom chips.
#. Data scientists and ML developers wishing to accelerate deep learning
performance.
#. New DL accelerator developers creating an end-to-end software stack from a
deep learning framework to their silicon.  




