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

      A machine learning environment, such as TensorFlow, MXNet, or
      neon.

   function graph

      The Intel nGraph library uses a function graph to represent an
      ``op``'s parameters and results.

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

   model description

      A description of a program's fundamental operations that are 
      used by a framework to generate inputs for computation. 

   export
   
      The serialized version of a trained model that can be passed to
      one of the nGraph backends for computation.      

