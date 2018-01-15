.. adding-new-ops.rst


Adding New Ops
##############

Overview
========
To add a new op in Intel® nGraph™ library, you'll need to do the following:

- Register the new op in ``op_graph``.
- Add adjoint function for computing gradients of the op (optional).
- Register op in transformer passes.
- Add implementation in transformers (such as CPU and GPU).
- Add corresponding tests.

Example
=======
In the following example, we'll walk though the steps for adding the ``Prod``
op. ``Prod`` computes the product of a tensor over given reduction axes. For
instance::

   import ngraph as ng
   axes = ng.make_axes([ng.make_axis(2), ng.make_axis(2)])
   x = ng.constant([[1., 2.], [3., 4.]], axes=axes)
   x_prod = ng.prod(x, reduction_axes=x.axes[1])
   print(ng.transformers.make_transformer().computation(x_prod)())
   # output:
   # [  2.  12.]

1. First, we need to register the new op in ``op_graph``. In general, an op is
   a sub-class of ``ngraph.op_graph.op_graph.Op``. To add a new op, we could
   inherit from the class ``Op`` or one of its descendant classes and implement
   required methods. We need to implement ``__init__()``, and if we
   want to define the derivative of the op, we need to implement
   ``generate_adjoints()``. For other advanced functionalities, please refer to
   the the source of ``ngraph.op_graph.op_graph.Op``.

   Let's look at ``DotOp`` for an example. ``DotOp`` inherits
   ``TensorOp`` which is a descendant of ``Op``. In the ``__init__()`` function,
   we do input arguments checking and set up the output axes in ``DotOp.axes``.::

        class DotOp(TensorOp):

              def __init__(self, x, y, **kwargs):
                 self.reduction_axes = x.axes & y.axes
                 self.x_out_axes = x.axes - self.reduction_axes
                 self.y_out_axes = y.axes - self.reduction_axes
                 self.bias = bias

                 axes = self.x_out_axes + self.y_out_axes

                 super(DotOp, self).__init__(
                     args=(x, y), axes=axes, **kwargs
                 )

            ...

2. Next, we could optionally add an adjoint function for computing gradients of
   the op in the ``generate_adjoints()`` function. There are two scenarios:

   a. If the gradients of the op can be represented by other ops available in
   ngraph, we could use those ops to implement the gradients.
   b. If that is not possible, or if we want to optimize the performance of the gradient
   computation, we could add new ops specifically for computing gradient.

   The ``generate_adjoints()`` function takes several arguments:

         - ``adjoints``: A dictionary that stores adjoints for the derivative being
           computed. The dictionary key is the op, and the value is the
           derivative of the op. ``adjoints`` needs to be accumulated by the
           ``generate_add_delta()`` function.
         - ``delta``: The back-propagated gradient from the downstream of the
           graph.
         - ``input_op_0``: The input operand to the op.
         - ``input_op_1, ...``: Other input operands to the op.
         - ``generate_adjoints()`` Takes variable-length inputs.

   Inside ``generate_adjoints()`` for each input op, we accumulate its
   gradients by calling ``input_op.generate_add_delta(adjoints, gradient_by_current_op)``.

   For more details on how autodiff works in Intel nGraph library, refer to the
   :ref:`autodiff <autodiff>` documentation.

   In this example, we could represent the gradients of the ``Prod`` by other
   Intel nGraph library ops, such as ``equal``, ``sum``, ``prod`` and ``broadcast``. Also,
   since we are using the ``create_reduction_op`` helper function, we define a
   ``prod_adjoints()`` function externally and pass it to the helper function.
   The helper function then applies it to the ``generate_adjoints()``
   in the generated ``Prod`` class.

   In ``ngraph/op_graph/op_graph.py``, we add ::

        def prod_adjoints(self, adjoints, delta, x):
            # axes
            axes = x.axes
            reduction_axes = self.reduction_axes

            # x_equal_zero
            x_equal_zero = equal(x, 0)

            # count 0's occurrence by reduction axes
            x_zero_count = sum(x_equal_zero, reduction_axes=reduction_axes)

            # create mask for zero count 0 and 1
            mask_zero = broadcast(equal(x_zero_count, 0), axes=axes)
            mask_one = broadcast(equal(x_zero_count, 1), axes=axes)

            # replace all 0 to 1
            x_replaced = equal(x, 0.) * 1. + (1. - equal(x, 0.)) * x

            # do product of x_replace and gradient
            x_replaced_prod = prod(x_replaced, reduction_axes=reduction_axes)
            x_replaced_grad = x_replaced_prod / x_replaced

            # multiply mask with mask for the two cases
            x_grad = mask_zero * x_replaced_grad + mask_one * x_equal_zero * x_replaced_grad

            x.generate_add_delta(
                adjoints,
                broadcast(delta, x.axes) * x_grad
            )

   Going back to the ``DotOp``: In its ``generate_adjoints`` function, we accumulate
   the gradients for the LHS operand ``x`` and RHS operand ``y`` respectively::

         class DotOp(TensorOp):
             ...

             def generate_adjoints(self, adjoints, delta, x, y):
                 x.generate_add_delta(
                     adjoints,
                     axes_with_order(dot(delta, y), x.axes)
                 )
                 y.generate_add_delta(
                     adjoints,
                     axes_with_order(dot(x, delta), y.axes)
                 )

3. The next step is to register the op in transformer passes. Transformer passes
   are used to simplify graph, to optimize ops for execution, and to meet device-specific constraints. 
   Some optimization passes are optional, while other passes could be required to ensure correctness. The two default passes we
   currently have are ``SimplePrune`` and ``CPUTensorShaping``. Refer to the :ref:`transformer passes <transformer_passes>` doc for more details.

   For ``Prod``, one of the optimizations we can do is that, if the tensor is
   filled with a constant value, we can replace ``Prod`` with the ``Power`` op.
   Therefore, in ``ngraph/transformers/passes/passes.py``, we add ::

        class SimplePrune(PeepholeGraphPass):
            ...

            @visit.on_type(Prod)
            def visit(self, op, x):
                """
                If x is filled with the same value, then replace the prod op
                with `power`.
                """
                if x.is_scalar and x.is_constant:
                    val = power(x.const, op.reduction_axes.size)
                    self.replace_op(op, constant(val))

4. Next, we need to add implementations of the op in transformers. Note that
   in the previous steps, we still haven't specified how the op will be executed
   (forward computation). In the current version of Intel nGraph library, the ops that are implemented in
   ``CPUTransformer`` and ``GPUTransformer`` are done by code generation for
   optimized performance.

   In ``ngraph/transformers/cputransform.py``, add the following for CPU
   code generation ::

        class CPUCodeGenerator(PyGen):
            ...

            @generate_op.on_type(Prod)
            def generate_op(self, op, out, x):
                self.append("np.prod({}, axis=0, out={})", x, out)

   In ``ngraph/transformers/gputransform.py``, add the following in the
   ``ElementWiseKernel`` class for the element-wise CUDA C kernel. Here, ops are
   first buffered in a list, and then the kernel is compiled at the end. ::

        class ElementWiseKernel(GPUKernel):
            ...

            @add_op.on_type(Prod)
            def add_op(self, op, out, x):
                self.add_reduction_op("prod", op, out, x)

   Finally in ``/ngraph/transformers/gpu/float_ew2.py`` add the following for
   the reduction op generation template. These are string templates for the
   generated CUDA C code. ::

        _redop_templates = {
            "prod": r"%(out)s = %(out)s * %(x)s;",
            ...
        }

        _redop32_templates = {
            "prod": r"%(out)s = %(out)s * __shfl_xor(%(out)s, i);",
            ...
        }

        _redop_inits = {
            "prod": "1.0f",
            ...
        }

5. The last step is to add the corresponding tests to verify the forward and
   backward computation. For ``ng.prod``, refer to the
   ``test_prod_constant()`` and ``test_prod_deriv`` test functions under
   ``tests/test_execution.py``.


