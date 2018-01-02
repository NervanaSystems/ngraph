.. tensors.rst: 

Tensors
#######

Tensors are created with ``AssignableTensorOp`` with several key attributes 
passed in as arguments:

- ``constant``: Immutable tensor that can be inlined.
- ``persistent``: Value is retained across computations (for example, weights).
- ``trainable``: Marker for a tensor that is meant to be retrieved as part of a 
  variable list.
- ``input``: Tensor that allows for a value to be passed in during runtime.

For convenience, we provide several helper functions to define commonly used 
tensor configurations, as shown below.

.. csv-table::
    :header: "Method", "constant", "persistent", "trainable", "input", "Example Usage"
    :widths: 20, 8, 8, 8, 8, 40
    :delim: |

    ``ng.constant`` | True | True | False | False | Constants specified during graph creation.
    ``ng.placeholder`` | False | True | False | True | Used for input values, typically from host.
    ``ng.persistent_tensor`` | False | True | False | False | Persistent tensors, such as velocity in SGD.
    ``ng.variable`` | False | True | True | False | Parameters that are updated during training.



Basic tensor descriptions
=========================

With Intel® nGraph™ library, we often have to reason about tensors before any 
computations or allocations are performed. For this reason, we use 
``tensor descriptions`` to hold enough metadata about tensors for analysis/
simplification. Basic tensor descriptions only have shape and element type 
information. Although the shape is an ordered list of lengths, the order does 
not imply a particular layout/striding for the dimensions. The basic tensor 
descriptions, with restrictions on dimensions and striding, are appropriate for 
the basic operations that all nGraph library transformers must implement. They 
might also be useful for front ends that describe tensors by shape.

If we know the layout of a tensor, we can compute the layout of subsequent 
slices and reshapings. But in nGraph library, we only know the layout for the 
subset of tensors whose layout has been explicitly provided by the frontend. But 
we still need information about which tensors are views of each other, dimension 
lengths, alignment constraints, slicing, etc. We use ``BasicTensorDescription`` 
to represent all the information the graph needs to know about tensors. This 
might vary during the transformation process. Little might be known about a 
tensor when it is first added to the graph, but by the time execution occurs, 
the tensor's layout needs to be known.


Tensor descriptions
-------------------

Describes a tensor by its shape and element type.

Attributes:
- dtype: The dtype of the elements.
- rank: The number of dimensions.
- read_only: *True* for an r-tensor, *False* for an l-tensor.
- shape: An n-tuple of non-negative integers. The length of the tuple is the rank.
- layout: strides and offset, if known.


Every basic tensor-valued ``Op`` corresponds to an r-tensor, meaning a tensor 
whose value can be on the right side of an assignment. Allocations correspond 
to l-tensors, which are tensors whose values can be assigned. Each tensor has a 
tensor description that describes the tensor and which is computed from the 
tensor descriptions of the parameters and arguments to the ``Op``.

During the transformation process, the tensor description can be augmented with 
additional information, such as a storage layout and storage assignment. The 
value of an ``Op`` might be a different view of a tensor, in which case the 
sharing must be indicated in its tensor description. An ``AssignableTensorOp`` 
is a special case of a tensor-valued ``Op`` in that its tensor is an l-tensor. 
At the end of the transformation process, all tensor descriptions for l-tensors 
must contain enough information for them to be allocated.

Implementation details
======================

Abstractly, an n-tensor is a map from an n-dimensional rectangle of non-negative 
integers to values of homogeneous type. In programming languages, there are two 
kinds of values, l-values and r-values. L-values can appear on the left side of 
an assignment and r-values can appear on the right side of an assignment. For 
example, ``x`` can be an l-value or an r-value, while ``x + y`` is an r-value. 
Likewise, if a tensor's values are l-values, the tensor is an l-tensor, and if 
the tensor's values are r-values, the tensors is an r-tensor. 

For example, the tensor ``x`` is an l-tensor since values can be assigned to its 
elements, as in ``x[...] = y``, while the tensor ``x + y`` is an r-tensor 
because values cannot be assigned to it. An r-tensor only needs to be able to 
provide values; it does not need to store them. The tensor ``x + y`` could 
produce the value for an index ``i`` by providing ``x[i] + y[i]`` every time it 
is needed, and a constant tensor could ignore the index and always produce the 
value.

Tensor ``y`` is a *simple view* of tensor ``x`` if there is some index 
translation function ``f`` such that ``y[i] == x[f(i)]`` for all valid indices 
of ``y``. Reshaping and slicing are two examples of tensor operations that 
create views. 

A  *complex view* involves multiple tensors and index translation functions. If 
``x[]`` is a list of tensors, and ``f[]`` a list of index translation functions, 
and there is a selection function ``s`` such that after setting 
``y[i] == x[s(i)][f[s(i)](i)]`` for all valid indices ``i`` of ``y`` and all 
values of ``x[...]``. In this case, ``s`` selects a tensor and index translation 
function for that tensor. Padding is an example of a complex view, in which a 
region of the values come from some other tensor, and the remaining values come 
from zero tensors. Transformers introduce views as they rewrite more abstract 
operations as simpler operations available on backends.

We can convert an r-tensor to an l-tensor by allocating an l-tensor for it 
initializing it with the r-values. If the values at particular indices are going 
to be used multiple times, this can reduce computation. Not all r-tensors should 
be converted to l-tensors. If  ``x`` and ``y`` are compatible 1-tensors, 
``x - y`` is a r-tensor. If we only want to compute the L2 norm of ``x - y`` we 
could use NumPy to compute the following:

.. code-block:: python

    def L2(x, y):
        t = x - y
        return np.dot(t.T, t)


Starting with the subtraction operation, NumPy first allocates a tensor for 
``t``. Every element in ``x``, ``y`` and ``t`` is touched once, and pages in 
``t`` are modified as elements are written in. Furthermore, accessing all the 
elements of ``x``, ``y``, and ``t`` can potentially evict other tensors from 
various CPU caches. 

Next, a view of ``t`` for ``t.T`` is allocated by NumPy. The memory footprint of 
a view is tiny compared to tensors. Computing the dot product accesses every 
element of ``t`` again. If ``t`` is larger than the memory cache, the recently 
cached elements near the end of ``t`` will be evicted so the ones near the 
beginning of ``t`` can be accessed. Also, because NumPy's dot operator does not 
function in place, it will also allocate another tensor for the output. 

When the function returns, the garbage collector sees that the view ``t.T`` and 
the tensor ``t`` are no longer referenced and will reclaim them. All the cache 
locations displaced by ``t`` are now unused. Furthermore, even though ``t`` is 
unallocated memory according the the heap, paging still sees it as modified 
pages. The page needs to be written back to paging before the physical memory 
can be given to other virtual memory. Likewise, the memory caches see the memory 
as modified and will need to invalidate caches for other cores.

Compare this with the following function:

.. code-block:: python

    def L2(x, y):
        s = 0
        for i in len(x):
            s = s + (x[i] - y[i])^2
        return s

As in the previous function, ``x`` and ``y`` will need to enter the cache, but 
there are no other tensors that need to be allocated, cached, and reclaimed, 
and there are no dirty pages to evict.


Dense L-Tensor Implementation
=============================

An L-tensor is typically represented as a contiguous region of memory and a 
mapping from the index to a non-negative integer offset into this memory. 
Essentially, every n-d tensor is a view of our memory, a 1-d linear tensor. An 
*l-value*, then, is the base address plus the index, adjusted for element size, 
and the *r-value* is the contents of the *l-value*. The `n-d` index mapping is 
characterized by an n-tuple of integers, called the stride, at an offset. The 
offset is added to the dot product of the strides and the n-tuple index to get 
the linear offset. If the linear tensor also has an n-tuple of integers, called 
the shape, bounds checking may be performed on the index. Sometimes it is 
important to align elements on particular memory boundaries. In this case, in 
addition to a shape we require an additional n-tuple called the size, which is 
greater than or equal to the shape to add padding for alignment.

There are many ways to map an index to a linear index that correspond to 
permutations of the stride n-tuple. Two common special cases are *row-major* 
and *column-major* ordering. 

In row-major ordering, the strides are listed in decreasing order and can be 
calculated using partial products of the allocated sizes for each dimension, 
multiplied from the right

For column-major ordering , the strides are in increasing order and are 
calculated by multiplying the sizes from the left. For example, if the sizes of 
the dimensions of a 3D-tensor are ``(5, 3, 2)``, then the row-major strides 
would be ``(6, 2, 1)`` and ``(1, 5, 15)`` for column major-order. 

.. Note::
   If two elements of the stride, shape, and size are permuted, then the same 
   linear index is given by permuting the index in the same way. For example, a 
   transpose view just requires these permutations.

Views allow for simpler implementation of tensor operations. For example, 
consider implementing a subtraction operation for arbitrary n-tensors of the 
same shape. For an implemented directory, an n-tuple index iterator would need 
to be maintained. However, if the n-tuple iterator would iterate over the 
linearized indices in the same order for both tensors, we can consider the 
*flattened* tensor view versions of these two tensors and use a single integer 
iterator to walk through pairs of elements from each tensor using the same 
offset for each. This produces the same result as if we had iterated through 
the two tensors using multidimensional indexing, but can result in the element 
pairs being accessed in different orders. This is only possible if the tensors 
have the same layout and strides.



