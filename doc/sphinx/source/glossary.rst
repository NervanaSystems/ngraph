:orphan:

.. glossary: 

Glossary 
========

.. glossary::

   function graph
	     The Intel nGraph library uses a function graph to represent an ``op``'s
	     parameters and results.

   op
      An op represents an operation. Ops are stateless and have zero or more 
      inputs and zero or more outputs. Some ops have additional constant 
      attributes. Every output of an op corresponds to a tensor and has an 
      element type and a shape. The element types and shapes of the outputs of 
      an op are determined by the inputs and attributes of the op.

   tensors
     Tensors are maps from *coordinates* to scalar values, all of the same type, 
     called the *element type* of the tensor.

   parameter
	    In the context of a function graph, a "parameter" refers to what "stands 
      in" for an argument in an ``op`` definition.

   result
       In the context of a function graph, the term "result" refers to what 
       stands in for the returned value.

   shape
       The shape of a tensor is a tuple of non-negative integers that represents an  
       exclusive upper bound for coordinate values.

   step
       An abstract "action" that produces zero or more tensor outputs from zero or more tensor 
       inputs. Steps correspond to *ops* that connect *nodes*.
           

