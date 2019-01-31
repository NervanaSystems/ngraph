:orphan:

.. glossary: 

Glossary 
========

.. glossary::
   :sorted:

   backend

      A component that can execute computations.

   bridge

      A component of nGraph that acts as a backend for a framework,
      allowing the framework to define and execute computations.

   data-flow graph

      Data-flow graphs are used to implement deep learning models. In  
      a data-flow graph, nodes represent operations on data and edges 
      represent data flowing between those operations. 

   framework

      Frameworks provide expressive user-facing APIs for constructing, 
      training, validating, and deploying DL/ML models: TensorFlow\*, 
      PaddlePaddle\*, MXNet\*, PyTorch\*, and Caffe\* are all examples of 
      well-known frameworks.

   function graph

      The Intel nGraph Library uses a function graph to represent an
      ``op``'s parameters and results.

   fusion
   
      Fusion is the fusing, combining, merging, collapsing, or refactoring
      of a graph's functional operations (``ops``) into one or more of
      nGraph's core ops.   

   ISA 

      An acronym for "Instruction Set Architecture," an ISA is machine code that  
      is compatible with the underlying silicon architecture. A realization of 
      an ISA is called an *implementation*. An ISA permits multiple 
      implementations that may vary in performance, physical size, memory use or 
      reuse, and monetary cost among other things. An ISA defines everything a 
      machine-language programmer needs to know in order to program a particular 
      backend device. What an ISA defines will differ among ISAs; in general, it
      defines things like:

         - supported *data types*; 
         - physical *states* available, such as the main memory and registers; 
         - *semantics*, such as the memory consistency and addressing modes; 
         - *low-level machine instructions* that comprise a machine language; 
         - and the *input/output model*.

      Be careful to not confuse ISAs with microarchitectures.    

   op

      An op represents an operation. Ops are stateless and have zero
      or more inputs and zero or more outputs. Some ops have
      additional constant attributes. Every output of an op
      corresponds to a tensor and has an element type and a shape. The
      element types and shapes of the outputs of an op are determined
      by the inputs and attributes of the op.

   parameter

      In the context of a function graph, a "parameter" refers to what
      "stands in" for an argument in an ``op`` definition.

   quantization

      Quantization refers to the conversion of numerical data into a lower-precision representation. Quantization is often used in deep learning to reduce the time and energy needed to perform computations by reducing the size of data transfers and the number of steps needed to perform a computation. This improvement in speed and energy usage comes at a cost in terms of numerical accuracy, but deep learning models are often able to function well in spite of this reduced accuracy. 

   result

      In the context of a function graph, the term "result" refers to
      what stands in for the returned value.

   shape

      The shape of a tensor is a tuple of non-negative integers that
      represents an exclusive upper bound for coordinate values.

   shared pointer

      The C++ standard template library has the template
      ``std::shared_ptr<X>``. A shared pointer is used like an ``X*``
      pointer, but maintains a reference count to the underlying
      object. Each new shared pointer to the object increases the
      count. When a shared pointer goes out of scope, the reference
      count is decremented, and, when the count reaches 0, the
      underlying object is deleted. The function template
      ``std::make_shared<X>(...)`` can be used similarly to ``new
      X(...)``, except it returns a ``std::shared_ptr<X>`` instead of
      an ``X*``.

      If there is a chain of shared pointers from an object back to
      itself, every object in the chain is referenced, so the
      reference counts will never reach 0 and the objects will never
      be deleted.

      If ``a`` referenced ``b`` and ``b`` wanted to track all
      references to itself and shared pointers were used both
      directions, there would be a chain of pointers form ``a`` to
      itself. We avoid this by using shared pointers in only one
      direction, and raw pointers for the inverse
      direction. ``std::enabled_shared_from_this`` is a class template
      that defines a method ``shared_from_this`` that provides a
      shared pointer from a raw pointer.

      nGraph makes use of shared pointers for objects whose lifetime
      is hard to determine when they are allocated.
   
   step

      An abstract "action" that produces zero or more tensor outputs
      from zero or more tensor inputs. Steps correspond to *ops* that
      connect *nodes*.
           
   tensors

      Tensors are maps from *coordinates* to scalar values, all of the
      same type, called the *element type* of the tensor.

      .. figure:: graphics/descriptor-of-tensor.png
         :width: 559px

   
   Tensorview 

      The interface backends implement for tensor use. When there are no more 
      references to the tensor view, it will be freed when convenient for the 
      backend.


   model description

      A description of a program's fundamental operations that are 
      used by a framework to generate inputs for computation. 

   export
   
      The serialized version of a trained model that can be passed to
      one of the nGraph backends for computation.

   NN

      :abbr:`NN (Neural Network)` is an acronym for "Neural Network". NN models 
      are used to simulate possible combinations of binary logic processing 
      and multi-layer (multi-dimensional) paths through which a :term:`data-flow graph` 
      may be mapped or computed. A NN does not have centralized storage; rather, 
      a NN manifests as information stored as patterns throughout the network 
      structure. NNs may be **Recurrent** (feedback loop) or **Nonrecurrent** 
      (feed-forward) with regard to the network vector.

   ANN

      :abbr:`Artificial Neural Network (ANN)`, often abbreviated as :term:`NN`. 

   RANN 

      :abbr:`Recurrent Artificial Neural Network (RANN)`, often abbreviated as 
      :term:`RNN`.


   RNN 
    
      A :abbr:`Recurrent Neural Network (RNN)` is a variety of :term:`NN` where 
      output nodes from a layer on a data-flow graph have loopback to nodes that 
      comprise an earlier layer. Since the RNN has no "centralized" storage, this 
      loopback is the means by which the ANN can "learn" or be trained. There are 
      several sub-categories of RNNs. The traditional RNN looks like: 

      :math:`s_t = tanh(dot(W,x_{t-1}) + dot(U, s_{t-1})`

      where :math:`x` is the input data, :math:`s` is the memory, and output is
      :math:`o_t = softmax(dot(V, s_t))`.  :doc:`ops/tanh`, :doc:`ops/dot`, and 
      :doc:`ops/softmax` are all nGraph :doc:`core Ops <ops/index>`.


   LSTM

      :abbr:`LSTM (Long Short-Term Memory)` is an acronym for "Long Short-Term 
      Memory". LSTMs extend on the traditional RNN by providing a number of ways 
      to "forget" the memory of the previous time step via a set of learnable 
      gates. These gates help avoid the problem of exploding or vanishing 
      gradients that occur in the traditional RNN.

   SGD

      :abbr:`Stochastic Gradient Descent (SGD)`, also known as incremental 
      gradient descent, is an iterative method for optimizing a 
      differentiable objective function.

   validated

      To provide optimizations with nGraph, we first confirm that a given 
      workload is "validated" as being functional; that is, we can
      successfully load its serialized graph as an nGraph :term:`function 
      graph`


 
