.. introduction:

############
Introduction
############


Developers working to craft solutions with :abbr:`Artificial Intelligence (AI)`
face a steep learning curve in taking their concepts from design to 
production. It can be challenging to create a :abbr:`Deep Learning (DL)` model 
that maintains a minimum standard of consistency, as it must be continually 
tweaked, adapted, or rewritten to use and optimize various parts of the stack 
during its life cycle. For DL models that do reach production-ready status, an 
entirely new set of problems emerges in how to scale and use larger and larger 
datasets, data that must be encrypted, data-in-motion, and of course, in 
finding the best compromises among speed, accuracy, and performance.  

Two general approaches to advancing deep learning performance dominate the 
industry today. The first is to design hardware dedicated exclusively to 
handling compute for specialized kinds of :abbr:`Machine Learning (ML)` or 
:abbr:`DL (Deep Learning)` operations; this approach essentially designs a 
custom network infrastructure *around* specific problems AI is supposed to 
solve. For example, many companies are actively developing specialized 
:abbr:`ASICs (Application Specific Integrated Circuits)` to speed-up 
training (one kind of ASIC) or to reduce inference latency (another kind 
of ASIC) in their cloud-based or local data centers. This approach works 
great for :abbr:`Cloud Service Providers (CSPs)` and others that have 
considerable budgets to invest in researching and building new hardware; 
however, it creates a significant burden on the developer who needs to 
invest in adapting the context of their model for training and then for 
inference, to figure out at least two data-cycle pipelines or deployment 
scenarios, and to decide what tradeoffs to make when and where.  

The second approach to making deep learning more efficient is to design a  
software stack that lets the :abbr:`Neural Network (NN)` adapt to whatever 
compute resources are available and deliver performance via software 
optimization. The nGraph Compiler stack is our solution to this second 
approach: it provides an inherently efficient graph-based compilation 
infrastructure designed to be compatible with many upcoming DL ASICs while 
also unlocking a massive performance boost on any existing hardware targets 
in a network, whether they are CPUs, GPUs, or other custom silicon. nGraph 
provides optimization opportunities at the graph level, where the 
network-to-device compilation can be managed with a series of “subgraphs” 
that can be handled in either a static or a dynamic manner. With such a 
graph-based neural network, it's possible to extract context semantics 
that make it easier to work with larger datasets, data that must be encrypted, 
and data-in-motion. Our solution also addresses the scalability issue with 
kernel libraries, the current popular solution to accelerating deep learning 
performance.  


Motivations
===========

The current state-of-the-art software solution for speeding up deep learning 
computation is to integrate kernel libraries like Intel® Math Kernel Library 
for Deep Neural Networks (Intel® MKL DNN) and Nvidia\*'s CuDNN into deep 
learning frameworks. These kernel libraries offer a runtime performance boost 
on specific hardware targets through highly-optimized kernels and other 
operator-level optimizations.

However, kernel libraries have three main problems: 

#. Kernel libraries do not support graph level optimizations.
#. Framework integration of kernel libraries does not scale.
#. There are too many kernels to write, and they require expert knowledge.

The nGraph Compiler stack is designed to address the first two problems. nGraph 
applies graph-level optimizations by taking computational graph from deep learning 
frameworks and reconstructing it with nGraph IR (Intermediate Representations). 
nGraph IR centralizes computational graphs from various frameworks and provides 
a unified way to connect backends for targeted hardware. To address the third 
problem, it integrates with PlaidML to generate code in LLVM, OpenCL, OpenGL, 
Cuda and Metal with low-level optimizations automatically applied. 

The following three sections explore the aformentioned three problems in more 
detail. 

1. Absence of graph-level optimizations
---------------------------------------

The diagram below illustrates a simple example of how a deep learning framework 
integrated with a kernel library is capable of running each operation in a 
computational graph optimally, but the graph itself may not be optimial: 

.. _figure-A:

.. figure:: ../graphics/intro_graph_optimization.png
   :width: 555px
   :alt: 

The graph is constructed to execute (A+B)*C, but we can further optimize the graph to be represented as A*C. From the first graph shown on the left, the operation on the constant B be can be computed at the compile time (known as constant folding), and the graph can be further simplified to the one on the right because the constant has value of zero. Without such graph level optimizations, a deep learning framework with a kernel library will compute all operations, and the resulting computation will be sub-optimal. 

2. Reduced scalability 
----------------------

Integrating kernel libraries into frameworks is increasingly becoming non-trivial due to growing number of new deep learning accelerators. For each new deep learning accelator, a kernel library must be developed by team of experts. This labor intensive work is further amplified by the number of frameworks as indicated in the following diagram with orange lines. 

.. _figure-B:

.. figure:: ../graphics/intro_kernel_to_fw_accent.png
   :width: 555px
   :alt: 
      
Each individual framework must be manually integrated with each hardware-specific kernel library. Each integration 
is unique to the framework and its set of deep learning operators, its view on 
memory layout, its feature set, etc. Each of these connections, then, represents 
significant work for what will ultimately be a brittle setup that is enormously 
expensive to maintain.  

nGraph solves this problem with nGraph bridges that connect to the deep learning frameworks. nGraph bridges take computational graphs from supported deep learning frameworks, and they reconstruct the graph using nGraph IR with a few primitive nGraph operations. With the unified computational graph, kernel libraries no longer need to be separately integrated to each deep learning frameworks. Instead, the libraries only need to support nGraph primitive operations, and this approach streamlines integration process for the backend.  

3. Increasing number of kernels 
-------------------------------

As mentioned in the pervious section, kernel libraries need to be integrated with multiple deep learning frameworks, and this arduous task becomes even harder due to increased numbers of required kernels for achieving optimial performance. The number of required kernels is product of number of chip designs, data types, operations, and the cardinality of each parameter for each operation. In the past, the number of required kernels was limited, but as the AI research and industry rapidly develops, the final product of required kernels is increasing exponentially. 

.. _figure-C:

.. figure:: ../graphics/intro_kernel_explosion.png
   :width: 555px
   :alt: 

PlaidML was designed to address the expoential growth of kernel needs. It takes two inputs: operation defined by the user and machine description of the targetted hardware. It utilizes a Domain Specific Language (DSL) called Tile which allows developers to express how an operation should calculate tensors in a intutitive mathematical form. PlaidML takes user defined Tile code along with targed machine description such as threads, max memory input, etc to automatically apply low level optimizations. This automated optimization does not require kernel libraries to be written and lifts heavy burden for kernel developers. It also provides flexibility to support newer deep learning models in absence of hand optimized kernels for the new operations.   

Our solution: nGraph and PlaidML
================================

We developed nGraph and integrated it with PlaidML to accelerate deep learning performance and address the scalabliity issue of kernel libraries. nGraph applies graph level optimization to deep learning computations and unifies computational graphs from deep learning frameworks with its IR to mitigate scalability problem for backends. 

PlaidML automatically applies low level deep learning performance optimizations in conjunction with nGraph's graph level optimizations. PlaidML also offers extensive support for many hardware targets with its ability to generate code in LLVM, OpenCL, OpenGL, CUDA, and Metal. 

nGraph and PlaidML thus provide best of both worlds. If there is a hardware backend with existing kernel libraries, nGraph can readily support the target hardware because the backend only needs to support a few nGraph primitive operations. If the hardware supports one of the PlaidML code generation languages, it can be programmed to execute deep learning computation by simply specifying machine description. 

This documentation provides technical details of nGraph's core functionality, and framework and backend integrations. Creating a compiler stack like nGraph and PlaidML requires expert knowledge, and we hope nGraph and PlaidML will lift burden for 
1. Framework owners needing to support new hardware
2. Data scientist and ML developers wishing to accelerate deep learning performance
3. New deep learning accelerator developers creating end-to-end software stack from deep learning frameworks to their silicon.  


