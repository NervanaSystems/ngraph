// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <sstream>

#include "op_graph.hpp"
#include "axes.hpp"
#include "util.hpp"

using namespace ngraph;

// def tensor_descriptions(args):
//     """
//     A list of tensor descriptions for Ops.

//     Arguments:
//       args: A list of Ops.

//     Returns:
//       A list of the Op's tensor descriptions.
//     """
//     return (arg.tensor_description() for arg in args)

// def tdcache():
//     """
//     Decorator to mark tensor description method as cached.

//     Returns:
//         Cache decorator set to use a particular cache.
//     """
//     return cachetools.cached(cache=tdcache.tensor_description_cache)

// tdcache.tensor_description_cache = {}

// @contextmanager
// def metadata(**metadata):
//     """
//     Capture all Ops created within the context. Hides ops created in this
//     context from parent contexts.
//     """
//     with Op.all_ops() as ops:
//         yield
//     for op in ops:
//         if isinstance(op, TensorValueOp):
//             # make sure tensorvalue op matches thing it reads from
//             op.metadata.update(op.states_read[0].metadata)
//         else:
//             op.metadata.update(metadata)

// def with_op_metadata(f, metadata=None):
//     """
//     Decorator to add metadata to all ops created inside the decorated function.
//     If this decorator is applied to a method of a class with a class
//     variable `metadata` defined as a dictionary then we add that to the
//     op metadata to attach.
//     """
//     metadata = metadata or dict()
//     assert isinstance(metadata, dict), "Metadata must be dict, not {}".format(type(metadata))

//     @wraps(f)
//     def wrapper(*args, **kwargs):
//         with Op.all_ops() as ops:
//             result = f(*args, **kwargs)
//         # If this decorator is applied to a method of a class with a class
//         # variable called `metadata` then we add that to the
//         if len(args) > 0 and hasattr(type(args[0]), 'metadata'):
//             metadata.update(type(args[0]).metadata)
//         for op in ops:
//             op.metadata.update(metadata)
//         return result
//     return wrapper

//================================================================================================
// DebugInfo
//================================================================================================

// class DebugInfo(object):
//     """Mixin that captures file/line location of an object's creation."""

//     def __init__(self, **kwargs):
//         # TODO This is a good first cut for debugging info, but it would be nice to
//         # TODO be able to reliably walk the stack back to user code rather than just
//         # TODO back past this constructor
//         super(DebugInfo, self).__init__(**kwargs)
//         frame = None
//         try:
//             frame = inspect.currentframe()
//             while frame.f_locals.get('self', None) is self:
//                 frame = frame.f_back
//             while frame:
//                 filename, lineno, function, code_context, index = inspect.getframeinfo(
//                     frame)
//                 if -1 == filename.find('ngraph/op_graph'):
//                     break
//                 frame = frame.f_back

//             self.filename = filename
//             self.lineno = lineno
//             self.code_context = code_context
//         finally:
//             del frame

//     @property
//     def file_info(self):
//         """
//         Return file location that created the node.

//         Returns:
//             String with file location that created the node.

//         """
//         return 'File "{filename}", line {lineno}'.format(
//             filename=self.filename, lineno=self.lineno)

//================================================================================================
// Op
//================================================================================================

Op::Op(std::vector<op_ptr> arg_list, op_ptr const_value, bool is_const)
    : NameableValue{"Op"}
{
    //     def __init__(self,
    //                  args=(),
    //                  metadata=None,
    //                  const=None,
    //                  constant=False,
    //                  persistent=False,
    //                  trainable=False,
    //                  **kwargs):
    //         super(Op, self).__init__(**kwargs)
    //         self._args = None
    //         self._set_args(as_op(arg) for arg in args)
    //         self.metadata = dict()

    //         if metadata is not None:
    //             if not isinstance(metadata, dict):
    //                 raise ValueError("Metadata must be of type dict,"
    //                                  "not {} of {}".format(type(metadata), metadata))
    //             self.metadata.update(metadata)

    //         # List to keep generation deterministic
    //         self._control_deps = OrderedSet()
    //         self._deriv_handler = None
    //         self._const = const
    //         self.uuid = uuid.uuid4()
    //         self._is_constant = constant
    //         self._is_persistent = persistent
    //         self._is_trainable = trainable

    //         # Add this op to the all op accounting lists
    //         ops = Op._get_thread_ops()[-1]
    //         if ops is not None:
    //             ops.append(self)
    //         all_ops = Op.get_all_ops()[-1]
    //         if all_ops is not None:
    //             all_ops.append(self)

    //         self.style = {}
    //         self._forward = None

    //         self.scope = None
}

// class Op(NameableValue):
//     """
//     Any operation that can be in an AST.

//     Arguments:
//         args: Values used by this node.
//         const: The value of a constant Op, or None,
//         constant (bool): The Op is constant.  Default False.
//         forward: If not None, the node to use instead of this node.
//         metadata: String key value dictionary for frontend metadata.
//         kwargs: Args defined in related classes.

//     Attributes:
//         const: The value of a constant.
//         constant (bool): The value is constant.
//         control_deps (OrderedSet): Ops in addtion to args that must run before this op.
//         persistent (bool): The value will be retained from computation to computation and
//             not shared.  Always True if reference is set.
//         metadata: Dictionary with of string keys and values used for attaching
//             arbitrary metadata to nodes.
//         trainable: The value is trainable.
//     """

//     # Default is to not collect Ops as they are created
//     @staticmethod
//     def _get_thread_ops():
//         """
//         :return: The stack of Ops being collected.
//         """
//         try:
//             ops = get_thread_state().ops
//         except AttributeError:
//             ops = [None]
//             get_thread_state().ops = ops
//         return ops

//     @staticmethod
//     def get_all_ops():
//         try:
//             all_ops = get_thread_state().all_ops
//         except AttributeError:
//             all_ops = [None]
//             get_thread_state().all_ops = all_ops
//         return all_ops

//     # We need to create another stack here because all_ops and captured_ops
//     # have different semantics that don't work with a shared stack
//     @staticmethod
//     @contextmanager
//     def all_ops(ops=None, isolate=False):
//         """
//         Collects all Ops created within the context. Does not hide ops created
//         in this context from parent contexts unless isolate is True.
//         """
//         if ops is None:
//             ops = []
//         try:
//             all_ops = Op.get_all_ops()
//             all_ops.append(ops)
//             yield (ops)
//         finally:
//             all_ops.pop()
//             parent = all_ops[-1]
//             if not isolate and parent is not None:
//                 parent.extend(ops)

//     @staticmethod
//     def all_op_references(ops):
//         """
//         Currently ops can have references to other ops anywhere in their __dict__, (not just args,
//         but the other typical places handled in serialization's `add_edges`). This function
//         iterates through an ops __dict__ attributes and finds all other ops recursively.

//         This is more powerful than the ordered_ops method which only considers args and
//         control_deps.
//         """
//         op_set = OrderedSet(ops)

//         def add_op(op):
//             op_set.add(op)
//             for key in op.__dict__:
//                 val = getattr(op, key)
//                 if isinstance(val, Op) and val not in op_set:
//                     add_op(val)
//                 elif isinstance(val, dict):
//                     for subkey in val:
//                         if isinstance(val[subkey], Op) and val[subkey] not in op_set:
//                             add_op(val[subkey])
//                 elif isinstance(val, (list, tuple, set, OrderedSet)):
//                     for item in val:
//                         if isinstance(item, Op) and item not in op_set:
//                             add_op(item)
//         for op in ops:
//             add_op(op)
//         return op_set

//     @staticmethod
//     def ordered_ops(roots):
//         """
//         Topological sort of ops reachable from roots. Notes ngraph is using
//         depenency edges rather than dataflow edges, for example,
//         `top_sort(a -> b -> c) => [c, b, a]`.

//         Args:
//             roots: List of ops.

//         Returns:
//             A list of sorted ops.
//         """
//         ordered_ops = []
//         available = OrderedSet()
//         counts = dict()
//         parents = defaultdict(OrderedSet)
//         ready = OrderedSet()

//         available.update(root.forwarded for root in roots)
//         while available:
//             node = available.pop()

//             if node in counts or node in ready:
//                 continue

//             children = OrderedSet((child.forwarded for child in node.all_deps))
//             if children:
//                 counts[node] = len(children)
//                 for child in children:
//                     parents[child].add(node)
//                 available.update(children)
//             else:
//                 ready.add(node)

//         while ready:
//             node = ready.pop()
//             ordered_ops.append(node)
//             for p in parents.get(node, []):
//                 count = counts[p] - 1
//                 if count == 0:
//                     ready.add(p)
//                     del counts[p]
//                 else:
//                     counts[p] = count
//         if len(counts) > 0:
//             raise ValueError("Graph not a DAG")

//         return ordered_ops

//     @staticmethod
//     def visit_input_closure(roots, fun):
//         """
//         Apply function `fun` in the topological sorted order of roots.

//         Args:
//             roots: List of ops.

//         Returns:
//             None
//         """
//         for op in Op.ordered_ops(roots):
//             fun(op)

//     def copy_with_new_args(self, args):
//         """
//         This method creates a new op given an original op and new args. The purpose here
//         is to replace args for an op with layout conversions as needed but keep the op the same
//         otherwise.
//         """
//         return (type(self))(*args)

//     def _set_args(self, args):
//         """
//         Internal function. Changes args.

//         Args:
//             args: The new arguments.

//         """
//         self._args = tuple(args)
//         self.invalidate_property_cache('all_deps')
//         self.invalidate_property_cache('call_info')

op_ptr Op::tensor()
{
    return forwarded;
}

//     @property
//     def effective_tensor_op(self):
op_ptr Op::effective_tensor_op()
{
    return forwarded;
}

//     @property
//     def states_read(self):
//         """

//         Returns: All state read by this op.

//         """
//         return OrderedSet()

//     @property
//     def states_written(self):
//         """

//         Returns: All state written by this op.

//         """
//         return OrderedSet()

//     def __str__(self):
//         return self.graph_label

//     def __repr__(self):
//         return '<{cl}({gl}):{id}>'.format(
//             cl=self.__class__.__name__,
//             gl=self.graph_label_type,
//             id=id(self)
//         )

//     @property
//     def is_constant(self):
//         """

//         Returns: True if this op is a constant tensor.

//         """
//         return False

//     @property
//     def const(self):
//         """

//         Returns: For a constant, returns the constant value.

//         """
//         return None

//     @property
//     def is_input(self):
//         """

//         Returns: True if this op is a tensor that the host can write to.

//         """
//         return False

//     @property
//     def is_persistent(self):
//         """

//         Returns: True if this op is a tensor whose value is preserved from computation
//             to computation.

//         """
//         return False

//     @property
//     def is_trainable(self):
//         """

//         Returns: True if this op is a tensor that is trainable, i.e. is Op.variables
//             will return it.

//         """
//         return False

//     @property
//     def is_placeholder(self):
//         """

//         Returns: True if this op is a placeholder, i.e. a place to attach a tensor.

//         """
//         return False

//     @property
//     def is_tensor_op(self):
//         """

//         Returns: True if this op is a tensor.

//         """
//         return False

//     @property
//     def is_scalar(self):
//         """

//         Returns: True if this op is a scalar.

//         """
//         return 0 == len(self.axes)

//     @property
//     def is_device_op(self):
//         """

//         Returns:
//             True if the Op executes on the device.
//         """
//         return True

//     @property
//     def scalar_op(self):
//         """
//         Returns the scalar op version of this op.  Will be overridden by subclasses
//         """
//         if not self.is_scalar:
//             raise ValueError()
//         return self

//     @property
//     def forward(self):
//         """
//         If not None, self has been replaced with forward.

//         When set, invalidates cached tensor descriptions.

//         Returns:
//              None or the replacement.
//         """
//         return self._forward

//     @forward.setter
//     def forward(self, value):
//         if value is self:
//             return

//         self.update_forwards()
//         value.update_forwards()

//         # Make sure everything that is supposed to happen
//         # before this op still happens
//         for dep in self._control_deps:
//             value.add_control_dep(dep)
//         self._forward = value
//         tdcache.tensor_description_cache.clear()
//         value.metadata.update(self.metadata)

//     @property
//     def forwarded(self):
//         """
//         Finds the op that handles this op.

//         Returns:
//              Follows forwarding to the op that should handle this op.
//         """
//         result = self
//         while True:
//             if result._forward is None:
//                 return result
//             result = result._forward

//     @cached_property
//     def all_deps(self):
//         """
//         Returns:
//             All dependencies of the op, including args and control_deps.
//             `x.all_deps == OrderedSet(x.args) | x.control_deps`, setter
//             functions are used to maintain this invariant. However, user
//             outside of the Op class should still avoid changing x._all_deps,
//             x._control_deps and x._args directly.
//         """
//         return OrderedSet(self.args) | self.control_deps

//     def invalidate_property_cache(self, property_name):
void Op::invalidate_property_cache(const std::string& property_name)
{
    // This is c++, not python. Ignore
}

//     @property
//     def args(self):
//         """All the inputs to this node."""
//         return self._args

//     @property
//     def control_deps(self):
//         """
//         Returns:
//             Control dependency of the op.
//         """
//         return self._control_deps

void Op::add_control_dep(op_ptr dep)
{
    //         dep = dep.forwarded
    //         if dep is not self and dep not in self.all_deps:
    if (!contains(control_deps, dep) && !contains(all_deps(), dep))
    {
        //             # update control_deps
        control_deps.insert(dep);
        //             self._control_deps.add(dep)
        //             # invalidate deps cache as self._control_deps is updated
        //             self.invalidate_property_cache('all_deps')
        invalidate_property_cache("all_deps");
    }
}

//     def remove_control_dep(self, dep):
//         """
//         Remove an op from the list of ops that need to run before this op.

//         Args:
//             dep: The op.

//         """
//         self.update_forwards()
//         if dep.forwarded in self.control_deps:
//             # update control_deps
//             self._control_deps.remove(dep.forwarded)
//             # invalidate deps cache as self._control_deps is updated
//             self.invalidate_property_cache('all_deps')

//     def update_forwards(self):
//         """
//         Replaces internal op references with their forwarded versions.

//         Any subclass that uses ops stored outside of args and all_deps
//         needs to override this method to update those additional ops.

//         This is mainly to reduce the number of places that need to explicitly check
//         for forwarding.

//         """
//         # replace self._args with self._args's forwarded op
//         args_forward = [op.forward for op in self.args]
//         if any(forward is not None for forward in args_forward):
//             new_args = tuple([op.forwarded for op in self.args])
//             if self._args != new_args:
//                 self._args = new_args
//                 self.invalidate_property_cache('all_deps')

//         # replace self._control_deps with self._control_deps's forwarded op
//         control_deps_forward = [op.forward for op in self.control_deps]
//         if any(forward is not None for forward in control_deps_forward):
//             new_control_deps = OrderedSet([op.forwarded for op in self.control_deps])
//             if self._control_deps != new_control_deps:
//                 self._control_deps = new_control_deps
//                 self.invalidate_property_cache('all_deps')

//     def replace_self(self, rep):
//         self.forward = as_op(rep)

//     @property
//     def deriv_handler(self):
//         """
//         Overrides processing of this op for this derivative.

//         Returns:
//             The op that should be used to process this op. If no deriv_handler has been set,
//             self is returned.

//         """
//         if self._deriv_handler is None:
//             return self
//         else:
//             return self._deriv_handler

//     @deriv_handler.setter
//     def deriv_handler(self, deriv_handler):
//         if deriv_handler is self:
//             deriv_handler = None
//         self._deriv_handler = deriv_handler

//     @property
//     def defs(self):
//         """
//         Returns:
//             For liveness analysis.  The storage associated with everything
//             in the returned list is modified when the Op is executed.

//         """
//         return [self]

//     def variables(self):
//         """
//         Return all trainable Ops used in computing this node.

//         Returns:
//             Set of trainable Ops.
//         """
//         return OrderedSet([op.tensor for op in Op.ordered_ops([self])
//                            if op.tensor.is_trainable])

//     def placeholders(self):
//         """
//         Return all placeholder Ops used in computing this node.

//         Returns:
//             Set of placeholder Ops.
//         """
//         return OrderedSet([op.tensor for op in Op.ordered_ops([self])
//                            if op.tensor.is_placeholder])

//     def tensor_description(self):
//         return None
tensor_description_ptr Op::tensor_description()
{
    return nullptr;
}

//     @cachetools.cached({})
//     def call_info(self):
//         """
//         Creates the TensorDescriptions (of this op or its arguments)
//         required to evaluate it.

//         The list is used to allocate buffers (in the transformers) and supply
//         values to the transform method (in the transform_call_info) method.

//         Only TensorDescriptions of the arguments are necessary.  A
//         TensorDescription of the output is generate by calling
//         self.tensor_description()
//         """
//         return list(tensor_descriptions(self.args))

//     @property
//     def is_commutative(self):
//         """

//         Returns: True if the Op is commutative.

//         """
//         return False

//     @property
//     def is_sequencing_op(self):
//         """

//         Returns:
//             True if this op's sole purpose is to influence the sequencing of other ops.

//         """
//         return False

//     @property
//     def is_state_op(self):
//         """

//         Returns:
//             True if this op is state.

//         """
//         return False

//================================================================================================
// MutateInsteadOfCopyWithNewArgsMixin
//================================================================================================

// class MutateInsteadOfCopyWithNewArgsMixin(object):
//     """
//     We cannot create new ops with new layouts for GPUCudaScatterSendOp and GPUCudaGatherSendOp.
//     The information available at this point is not sufficient to create them (issue #1410).

//     """
//     def __init__(self, **kwargs):
//         super(MutateInsteadOfCopyWithNewArgsMixin, self).__init__(**kwargs)

//     def copy_with_new_args(self, args):
//         self._set_args(args)
//         return self

// def as_ops(xs):
//     """
//     Converts an iterable of values to a tuple of Ops using as_op.

//     Arguments:
//         xs: An iterable of values.

//     Returns:
//         A tuple of Ops.
//     """
//     return tuple(as_op(x) for x in xs)

//================================================================================================
// AssignOp
//================================================================================================

// class AssignOp(Op):
//     """
//     tensor[...] = val.

//     Arguments:
//         tensor (AssignableTensorOp): An assignable TensorOp.
//         val: The value to assign.
//         **kwargs: Args for related classes.
//     """

//     def __init__(self, tensor, val, **kwargs):
//         # convert val to op
//         # TODO: requires explicit broadcast in future
//         if not isinstance(val, Op):
//             val = as_op(val)
//             if len(val.axes) == len(tensor.axes):
//                 val = cast_axes(val, tensor.axes)

//         # automatic broadcast
//         # currently requires val's axes to be a subset of tensor's axes
//         # TODO: requires explicit broadcast in future
//         if len(val.axes - tensor.axes) > 0:
//             raise ValueError(
//                 "tensor(LHS) has axes %s, val(RHS) has axes %s,"
//                 "val's axes should be subset of tensor's axes" %
//                 (val.axes, tensor.axes))
//         val = broadcast(val, tensor.axes)

//         super(AssignOp, self).__init__(args=(tensor, val), **kwargs)

//     @property
//     def states_written(self):
//         return self.args[0].states_read

//     @property
//     def states_read(self):
//         return self.args[1].states_read

//================================================================================================
// AssignOneDOp
//================================================================================================

// class AssignOneDOp(Op):
//     """
//     Assign a value to a 1d tensor.

//     Arguments:
//         tensor (AssignableTensorOp): The value to assign to.
//         value (TensorOp): The value.
//     """

//     def __init__(self, tensor, val, **kwargs):
//         if val.is_scalar:
//             val = val.scalar_op
//         super(AssignOneDOp, self).__init__(args=(tensor, val), **kwargs)

//     @property
//     def states_written(self):
//         return self.args[0].states_read

//     @property
//     def states_read(self):
//         return self.args[1].states_read

// def assign(lvalue, rvalue):
//     """
//     Assignment; lvalue <= rvalue

//     Arguments:
//         lvalue: Tensor to assign to.
//         rvalue: Value to be assigned.
//         item (optional):
//     """
//     return AssignOp(lvalue, rvalue)

// def set_item(tensor, item, value):
//     shape = tensor.tensor_description().shape
//     for sl, l in zip(item, shape):
//         if not isinstance(sl, slice):
//             sl = slice(sl)
//         start, end, step = sl.indices(l)
//         if step <= 0:
//             raise ValueError('Invalid slice (negative step) in item {}'.format(item))
//     return assign(tensor_slice(tensor, item, axes=value.axes), value)

//================================================================================================
// ControlBlockOp
//================================================================================================

// class ControlBlockOp(Op):
//     """
//     An Op that affects execution sequencing.
//     """
//     def __init__(self, **kwargs):
//         super(ControlBlockOp, self).__init__(**kwargs)

//     @property
//     def is_device_op(self):
//         """

//         Returns:
//             False, because this is handled by the transformer.
//         """
//         return False

//================================================================================================
// ParallelOp
//================================================================================================
ParallelOp::ParallelOp(const std::vector<op_ptr>& ops)
    : Op{ops}
{
    for (auto op : ops)
    {
        add_control_dep(op);
    }
}

parallel_op_ptr doall(const std::vector<op_ptr>& ops)
{
    return std::make_shared<ParallelOp>(ops);
}

//================================================================================================
// ComputationOp
//================================================================================================

// class ComputationOp(ParallelOp):
//     """
//     Represents a host-callable graph computation.

//     Arguments:
//         returns: Values returned by the computation. A list, set, or op.
//         *args: Inputs to the computation. Must be placeholders or variables.

//     Parameters:
//         returns: Ops returned.
//         parameters: Parameter ops.
//     """
//     def __init__(self, returns, *args, **kwargs):
//         if isinstance(returns, collections.Container):
//             all = type(returns)(as_op(ret) for ret in returns)
//         elif isinstance(returns, Op):
//             all = [as_op(returns)]
//         elif returns is not None:
//             raise ValueError()
//         else:
//             all = []

//         self.values = all
//         self.returns = returns
//         super(ComputationOp, self).__init__(all=all, **kwargs)

//         def is_input(arg):
//             return arg.tensor.is_input

//         placeholders = self.placeholders()
//         if len(args) == 1 and args[0] == 'all':
//             args = placeholders

//         args = tuple(as_op(arg) for arg in args)
//         arg_tensors = set(arg.tensor for arg in args)
//         missing_tensors = [t for t in placeholders - arg_tensors]
//         if len(missing_tensors) > 0:
//             raise ValueError(("All used placeholders must be supplied to a "
//                               "computation. Currently missed {}."
//                               ).format(missing_tensors))

//         for arg in args:
//             if not (arg.tensor.is_input):
//                 raise ValueError((
//                     'The arguments to a computation must all be Ops with property '
//                     'is_input=True, but the op passed had is_input=False.'
//                     'In most cases you want to pass placeholder ops in as arguments.  '
//                     '{op} was passed in, of type {op_type}.'
//                 ).format(
//                     op=arg,
//                     op_type=arg.__class__.__name__,
//                 ))

//         self.parameters = args
//         for arg in args:
//             self.add_control_dep(arg)

// def computation(returns, *args):
//     """
//     Defines a host-callable graph computation.

//     Arguments:
//         returns: Values returned by the computation. A list, set, or op.
//         *args: Inputs to the computation.

//     Returns:
//         A computation op.
//     """

//     return ComputationOp(returns, *args)

//================================================================================================
// Fill
//================================================================================================

// class Fill(Op):
//     """
//     Fill a tensor with a scalar value.

//     Arguments:
//         tensor (AssignableTensorOp): An assignable TensorOp.
//         scalar: A scalar value.
//     """

//     def __init__(self, tensor, scalar, **kwargs):
//         super(Fill, self).__init__(args=(tensor,), **kwargs)
//         if isinstance(scalar, TensorOp):
//             if scalar.is_constant:
//                 scalar = scalar.const
//             else:
//                 raise ValueError("{} is not a scalar constant".format(scalar))
//         else:
//             npscalar = np.asarray(scalar, dtype=tensor.dtype)
//             if 0 != len(npscalar.shape):
//                 raise ValueError("{} is not a scalar".format(scalar))
//             scalar = npscalar[()]

//         self.scalar = scalar

//     @property
//     def states_written(self):
//         return self.args[0].states_read

//     @property
//     def states_read(self):
//         return self.args[0].states_read

// def fill(x, scalar):
//     return Fill(x, scalar)

//================================================================================================
// TensorOp
//================================================================================================

// class TensorOp(Op):
//     """
//     Super class for all Ops whose value is a Tensor.

//     Arguments:
//         axes: The axes of the tensor.
//         dtype: The element type of the tensor.
//         scale: If specified, a scaling factor applied during updates.
//         is_value_op: If specified, the normal dtype/axes/scale defaulting is disabled
//           since those values will be supplied by a subclass, such as ValueOp.
//         **kwargs: Arguments for related classes.
//     """
TensorOp::TensorOp(std::vector<op_ptr> args, std::vector<Axis> axes, float scale, bool is_value_op)
    : Op(args)
{
    //     def __init__(self, dtype=None, axes=None, scale=None, is_value_op=None, **kwargs):
    //         super(TensorOp, self).__init__(**kwargs)
    //         if not is_value_op:
    //             self.dtype = default_dtype(dtype)
    //             if axes is not None:
    //                 axes = make_axes(axes)
    //             self._axes = axes
    //             self.scale = scale

    //     @property
    //     def is_tensor_op(self):
    //         return True

    //     @property
    //     @cachetools.cached(cache=dict())
    //     def one(self):
    //         """
    //         Returns a singleton constant 1 for this Op. Used by DerivOp to ensure that
    //          we don't build unique backprop graphs for every variable.

    //         Returns:
    //             A unique constant 1 associated with this TensorOp.

    //         """
    //         return as_op(1)

    //     @cachetools.cached({})
    //     def adjoints(self, error):
    //         """
    //         Returns a map containing the adjoints of this op with respect to other
    //         ops.

    //         Creates the map if it does not already exist.

    //         Arguments:
    //             error (TensorOp, optional): The tensor holding the error value
    //                 the derivative will be computed at. Must have the same axes as dependent.

    //         Returns:
    //             Map from Op to dSelf/dOp.
    //         """
    //         adjoints = {
    //             self.tensor: error,
    //         }

    //         # visit ops in reverse depth first post-order. it is important that
    //         # ordered_ops returns a copy of this traversal order since the graph
    //         # may change as we generate adjoints and we don't want to visit those
    //         # new ops. Some ops may be containers for other ops, so we create an
    //         # ordered set to ensure we don't do multiple backprops.
    //         processed = set()
    //         for o in reversed(Op.ordered_ops([self])):
    //             if o.tensor in processed:
    //                 continue
    //             if o.tensor in adjoints:
    //                 adjoint = adjoints[o.tensor]
    //                 if o.scale is not None:
    //                     adjoint = adjoint * o.scale

    //                 deriv_handler = o.deriv_handler

    //                 # find hetr distribution metadata, pass other data if exists
    //                 # todo add reduce func metadata key when fixed #1436
    //                 hetr_meta_key = ['device', 'device_id', 'parallel']
    //                 hetr_metadata = {k: o.metadata[k] for k in hetr_meta_key
    //                                  if o.metadata.get(k) is not None}
    //                 with metadata(**hetr_metadata):
    //                     deriv_handler.generate_adjoints(adjoints, adjoint, *deriv_handler.args)

    //                 processed.add(o.tensor)

    //         return adjoints

    //     def generate_add_delta(self, adjoints, delta):
    //         """
    //         Adds delta to the backprop contribution..

    //         Arguments:
    //             adjoints: dy/dOp for all Ops used to compute y.
    //             delta: Backprop contribute.
    //         """
    //         if not self.axes.is_equal_set(delta.axes):
    //             raise ValueError(
    //                 'delta axes {} do not match adjoint axes {}'
    //                 .format(delta.axes, self.axes)
    //             )
    //         if self not in adjoints:
    //             adjoints[self] = delta
    //         else:
    //             adjoints[self] = delta + adjoints[self]

    //     # Magic methods for builtin operations we want to use for creating nodes
    //     def __neg__(self):
    //         return negative(self)

    //     def __pos__(self):
    //         return self

    //     def __abs__(self):
    //         return absolute(self)

    //     def __add__(self, val):
    //         return add(self, val)

    //     def __radd__(self, val):
    //         return add(val, self)

    //     def __sub__(self, val):
    //         return subtract(self, val)

    //     def __rsub__(self, val):
    //         return subtract(val, self)

    //     def __mul__(self, val):
    //         return multiply(self, val)

    //     def __rmul__(self, val):
    //         return multiply(val, self)

    //     def __div__(self, val):
    //         return divide(self, val)

    //     def __mod__(self, val):
    //         return mod(self, val)

    //     def __truediv__(self, val):
    //         return divide(self, val)

    //     def __rtruediv__(self, val):
    //         return divide(val, self)

    //     def __floordiv__(self, val):
    //         return floordivide(self, val)

    //     def __rdiv__(self, val):
    //         return divide(val, self)

    //     def __pow__(self, val):
    //         return power(self, val)

    //     def __rpow__(self, val):
    //         return power(val, self)

    //     # Python always uses eq for comparing keys, so if we override __eq__ we
    //     # cannot have sets of tensors, or using them as dictionary keys.  So,
    //     # we must use Equal explicitly in transform.  defmod and define __eq__
    //     # if it can ensure that its nodes do not need to be used as keys.
    //     # def __eq__(self, val):
    //     #    return equal(self, val)

    //     # def __ne__(self, val):
    //     #    return not_equal(self, val)

    //     def __lt__(self, val):
    //         return less(self, val)

    //     def __gt__(self, val):
    //         return greater(self, val)

    //     def __le__(self, val):
    //         return less_equal(self, val)

    //     def __ge__(self, val):
    //         return greater_equal(self, val)

    //     def __setitem__(self, key, val):
    //         if key == slice(None) or key is Ellipsis:
    //             return assign(self, val)
    //         raise ValueError("Setting {} is not supported yet".format(key))

    //     # Only works when capturing ops
    //     def __iadd__(self, val):
    //         return assign(self, self + val)

    //     # Only works when capturing ops
    //     def __isub__(self, val):
    //         return assign(self, self - val)

    //     # Only works when capturing ops
    //     def __imul__(self, val):
    //         return assign(self, self * val)

    //     # Only works when capturing ops
    //     def __idiv__(self, val):
    //         return assign(self, self / val)

    //     def __getitem__(self, item):
    //         if isinstance(item, slice) and len(self.axes) > 1:
    //             item = (item,)
    //         item += tuple(slice(None) for _ in range(len(self.axes) - len(item)))
    //         return tensor_slice(self, item)

    //     def __axes__(self):
    //         return self.axes

    //     @tdcache()
    //     def tensor_description(self):
    //         """
    //         Returns a TensorDescription describing the output of this TensorOp

    //         Returns:
    //           TensorDescription for this op.
    //         """
    //         if "layout" in self.metadata:
    //             return TensorDescription(self.axes,
    //                                      op=self,
    //                                      layout=self.metadata["layout"],
    //                                      dtype=self.dtype,
    //                                      is_persistent=self.is_persistent,
    //                                      is_input=self.is_input,
    //                                      is_placeholder=self.is_placeholder)
    //         else:
    //             return TensorDescription(self.axes, dtype=self.dtype, name=self.name,
    //                                      op=self,
    //                                      is_persistent=self.is_persistent,
    //                                      is_input=self.is_input,
    //                                      is_placeholder=self.is_placeholder)

    //     @property
    //     def axes(self):
    //         """

    //         Returns: The axes of the tensor.

    //         """
    //         if self._axes is not None:
    //             return self._axes
    //         else:
    //             raise NotImplementedError

    //     @axes.setter
    //     def axes(self, value):
    //         if self._axes is not None:
    //             raise ValueError()
    //         self._axes = value

    //     @property
    //     def has_axes(self):
    //         """

    //         Returns: True if axes have been set.

    //         """
    //         return self._axes is not None

    //     def insert_axis(self, index, axis):
    //         """
    //         Inserts an axis
    //         Arguments:
    //             index   : Index to insert at
    //             axis    : The Axis object to insert
    //         """
    //         if self._axes is None:
    //             raise ValueError()
    //         self._axes.insert(index, axis)

    //     def append_axis(self, axis):
    //         if self._axes is None:
    //             raise ValueError()
    //         self._axes.append(axis)

    //     def generate_adjoints(self, adjoints, delta, *args):
    //         """
    //         With delta as the computation for the adjoint of this Op, incorporates delta into the
    //         adjoints for thr args.

    //         Args:
    //             adjoints: dy/dOp for all ops involved in computing y.
    //             delta: Backprop amount for this Op.
    //             *args: The args of this Op.
    //         """
    //         pass

    //     @property
    //     def shape(self):
    //         """
    //         This is required for parameter initializers in legacy neon code.  It
    //         expects layers to implement a shape that it can use to pass through
    //         layers.

    //         Returns: self.axes
    //         """
    //         return self.axes

    //     def shape_dict(self):
    //         """
    //         Retuns: shape of this tensor as a dictionary
    //         """
    //         return self.axes.shape_dict()

    //     def mean(self, reduction_axes=None, out_axes=None):
    //         """
    //         Used in Neon front end.

    //         Returns: mean(self)

    //         """
    //         return mean(self, reduction_axes=reduction_axes, out_axes=out_axes)
}

//================================================================================================
// ValueOp
//================================================================================================

// class ValueOp(TensorOp, ControlBlockOp):
//     """
//     Mixin class for ops whose value is another op.
//     Arguments:
//         tensor: The tensor supplying the value for this op.
//     """
ValueOp::ValueOp(op_ptr t)
    : TensorOp()
    , tensor{std::static_pointer_cast<AssignableTensorOp>(t)}
{
    //     def __init__(self, tensor=None, **kwargs):
    //         super(ValueOp, self).__init__(args=(), is_value_op=True, **kwargs)
    //         self._tensor = tensor
}

//     def tensor_description(self):
//         return self.tensor.tensor_description()

//     @property
//     def tensor(self):
//         """
//         The op that ultimately supplies the value. See value_tensor.
//         Returns:
//             The op that supplies the value.
//         """
//         if self._tensor is not None:
//             return self._tensor.forwarded.tensor.forwarded
//         else:
//             return None

//     @property
//     def value_tensor(self):
//         """
//         The op whose value is returned by this op.
//         Returns:
//             The immediate value returned by this op; see tensor for the closure.
//         """
//         if self._tensor is not None:
//             return self._tensor.forwarded
//         else:
//             return None

//     @value_tensor.setter
//     def value_tensor(self, tensor):
//         self._tensor = tensor

//     @property
//     def all_deps(self):
//         """
//         TODO: use cached property as other Op
//         """
//         base_deps = super(ValueOp, self).all_deps
//         if self.value_tensor is not None and self.value_tensor.is_device_op:
//             # Add value_tensor if it is a real op
//             return base_deps | OrderedSet([self.value_tensor])
//         else:
//             return base_deps

//     @property
//     def is_tensor_op(self):
//         return self.tensor.is_tensor_op

//     @property
//     def axes(self):
//         return self.tensor.axes

//     @property
//     def dtype(self):
//         return self.tensor.dtype

//     @dtype.setter
//     def dtype(self, dtype):
//         self.tensor.dtype = default_dtype(dtype)

//     @property
//     def is_constant(self):
//         return self.tensor.is_constant

//     @property
//     def const(self):
//         return self.tensor.const

//     @property
//     def scale(self):
//         return self.tensor.scale

//     @property
//     def states_read(self):
//         return self.value_tensor.states_read

//     @property
//     def states_written(self):
//         return self.value_tensor.states_written

//     def generate_add_delta(self, adjoints, delta):
//         self.tensor.generate_add_delta(adjoints, delta)

//     @property
//     def effective_tensor_op(self):
//         # Due to hard to correct class hierarchy, state access is wrapped in ValueOp, but we
//         # always want state access wrapped in a state reader such as TensorValueOp, so we
//         # need to resort to some ugliness here.
//         tensor = self._tensor
//         if tensor.is_state_op:
//             return self.forwarded
//         return tensor.effective_tensor_op

//================================================================================================
// SequentialOp
//================================================================================================

// class SequentialOp(ValueOp):
//     """
//     Given a list of ops, ensure that every op that has not already been executed is executed in
//     the given order. The value of the last op is the value of this op.

//     Ops will only be executed once, so to return the value of an earlier op, just add it again at
//     the end of the list.

//     Control dependencies are not computed until after the graph is computed, i.e. after derivatives
//     are expanded.

//     Arguments:
//         ops: Sequence of ops to compute. If not specified, set the attribute ops when known. This
//             is useful for subclassing.

//     Attributes:
//         ops: The list of ops to be computed. The last op is the returned value.
//     """
//     def __init__(self, ops=None, **kwargs):
//         super(SequentialOp, self).__init__(**kwargs)
//         self.value_tensor = None
//         self._ops = None
//         if ops is not None:
//             self.ops = ops

//     @property
//     def ops(self):
//         return self._ops

//     @ops.setter
//     def ops(self, ops):
//         self._ops = list(as_op(op).forwarded for op in ops)

//         for op in self._ops:
//             self.add_control_dep(op)
//         self.value_tensor = self._ops[-1]

//         # Ops that have already executed.
//         done_ops = set()

//         # State => op_tops that have written state
//         writers = defaultdict(OrderedSet)
//         # State => op_tops that have read state
//         readers = defaultdict(OrderedSet)
//         for op_top in self._ops:
//             ordered_ops = Op.ordered_ops([op_top])
//             # Make ops that read/write state execute after the op_tops that last read/wrote
//             # the state.
//             for op in ordered_ops:
//                 if op in done_ops:
//                     # The op already ran, so it doesn't run here
//                     continue
//                 for state in op.states_read:
//                     for write_op in writers[state]:
//                         op.add_control_dep(write_op)
//                 for state in op.states_written:
//                     for read_op in readers[state]:
//                         op.add_control_dep(read_op)
//             # Register this op_top with each state it read/wrote.
//             for op in ordered_ops:
//                 if op in done_ops:
//                     # The op already ran, so it doesn't run here
//                     continue
//                 for state in op.states_written:
//                     writers[state].add(op_top)
//                 for state in op.states_read:
//                     readers[state].add(op_top)
//             done_ops.update(ordered_ops)

//     @property
//     def is_sequencing_op(self):
//         """

//         Returns:
//             True if this op's sole purpose is to influence the sequencing of other ops.

//         """
//         return True

// def sequential(ops=None):
//     """
//     Compute every op in order, compatible with existing dependencies, returning last value.

//     Ops will only be executed once, so to return the value of an earlier op, just add it again at
//     the end of the list.

//     Arguments:
//         ops: Sequence of ops to compute.
//     """
//     sequential_op = SequentialOp(ops)
//     sequential_op.deriv_handler = sequential_op.value_tensor
//     # Note: Can't return value_tensor here because we may need some ops to execute
//     # after it. For example,
//     # op_1, op_2, op_3, op_1 has value of op_1, but op_1 won't force op_2 and op_3 to run.
//     return sequential_op

//================================================================================================
// TensorValueOp
//================================================================================================

// class TensorValueOp(ValueOp):
//     """
//     A read of an AssignableTensorOp.

//     This provides a way to maintain different control information on different
//     versions of state.

//     Arguments:
//         tensor: The tensor being wrapped.
//     """
TensorValueOp::TensorValueOp(op_ptr tensor)
    : ValueOp(tensor)
{
}
//     def __init__(self, tensor, **kwargs):
//         super(TensorValueOp, self).__init__(tensor=tensor, **kwargs)
//         for key in ['device', 'device_id', 'parallel']:
//             if key in tensor.metadata:
//                 self.metadata[key] = tensor.metadata[key]

//     @property
//     def states_read(self):
//         return OrderedSet([self.tensor])

//     @property
//     def effective_tensor_op(self):
//         return self.forwarded

//================================================================================================
// PatternLabelOp
//================================================================================================

// class PatternLabelOp(TensorOp):
//     """
//     An op to represent label in the pattern to be matched in graph

//     constraint_fn is a predicate that must hold in order to bind the
//     label to its matching op. By default, constraint_fn is always true.

//     """
//     def __init__(self, label, constraint_fn=(lambda op: True), axes=None, **kwargs):
//         if axes is None:
//             axes = {}
//         super(PatternLabelOp, self).__init__(axes=axes, **kwargs)
//         self.label = label
//         self.constraint_fn = constraint_fn

//================================================================================================
// PatternSkipOp
//================================================================================================

// class PatternSkipOp(TensorOp):
//     """
//     An op to allow user of pattern matching to skip match for certain ops

//     is_optional_op_fn is a predicate that must be defined to specify
//     optional ops. By default, is_optional_op_fn is false.

//     """
//     def __init__(self, arg, is_optional_op_fn=(lambda op: False), **kwargs):
//         super(PatternSkipOp, self).__init__(axes={}, args=(arg,), **kwargs)
//         self.is_optional_op_fn = is_optional_op_fn

//================================================================================================
// IndexOp
//================================================================================================

// class IndexOp(with_metaclass(abc.ABCMeta, TensorOp)):
//     """
//     An base class for ops that change how a tensor is indexed; i.e. get a view of the same tensor.

//     Arguments:
//         x: A view of a tensor.

//     Returns:
//         A view of the tensor.
//     """
IndexOp::IndexOp(op_ptr op, Axes axes)
    : TensorOp()
{
    //     def __init__(self, x, **kwargs):
    //         super(IndexOp, self).__init__(
    //             args=(x,),
    //             dtype=x.dtype,
    //             **kwargs
    //         )
}

//     @abc.abstractmethod
//     def transform_tensor_description(self, tensor_description):
//         """
//         Apply this index operation to tensor_description.

//         Args:
//             tensor_description: TensorDescription of the input view.

//         Returns:
//             TensorDescription of the transformed view.

//         """

//     @tdcache()
//     def tensor_description(self):
//         return self.transform_tensor_description(self.args[0].tensor_description()).named(
//             self.name)

//     @property
//     def is_scalar(self):
//         """
//         Reshape adds shape information, but we retain being a scalar.

//         Returns:
//             True if the value comes from a scalar.

//         """
//         return self.args[0].is_scalar

//     @property
//     def scalar_op(self):
//         return self.args[0].scalar_op

//     @property
//     def is_device_op(self):
//         """
//         Returns:
//             False, because this is handled by the transformer.
//         """
//         return False

//     @property
//     def states_read(self):
//         # Reshapes are views of the underlying tensor, so the states are the same.
//         return self.args[0].states_read

//================================================================================================
// Transpose
//================================================================================================

// class Transpose(IndexOp):
//     """
//     Used to reverse the axes of a tensor.

//     Arguments:
//         x: A tensor.
//     """

//     def __init__(self, x, **kwargs):
//         super(Transpose, self).__init__(
//             x,
//             axes=reversed(x.axes),
//             **kwargs
//         )

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.transpose()

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, Transpose(delta))

//================================================================================================
// AxesCastOp
//================================================================================================

// class AxesCastOp(IndexOp):
//     """
//     Used to label a tensor with known axes, without altering its value

//     Arguments:
//         x: A tensor.
//         axes: The new axes.
//     """

//     def __init__(self, x, axes, **kwargs):
//         axes = make_axes(axes)
//         self._check_valid_axes(x, axes)
//         super(AxesCastOp, self).__init__(x, axes=axes, **kwargs)

//     def _check_valid_axes(self, x, axes):
//         if not x.is_scalar and x.axes.lengths != axes.lengths:
//             raise ValueError("casting axes {} must have the same length as original axes {}"
//                              .format(axes, x.axes))

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.cast(self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, cast_axes(delta, x.axes))

//================================================================================================
// RoleCastOp
//================================================================================================

// class RoleCastOp(AxesCastOp):
//     """
//     Used to set the names of the axes of a tensor, without altering its value.
//     If the names of the new axes are the same as the incoming tensor's axes,
//     leave the original axis alone.  Otherwise, create a new axis with the
//     length of the original and the name of the new.
//     Arguments:
//         x: A tensor.
//         axes: The new axes.
//     """

//     def __init__(self, x, axes, **kwargs):
//         axes = make_axes([
//             old_axis if old_axis == new_axis else make_axis(old_axis.length, new_axis.name)
//             for old_axis, new_axis in zip(x.axes, axes)
//         ])
//         self._check_valid_axes(x, axes)

//         super(RoleCastOp, self).__init__(x, axes=axes, **kwargs)

//     def _check_valid_axes(self, x, axes):
//         if len(x.axes) != len(axes):
//             raise ValueError(
//                 "casting axes {} must have the same number of axes as original axes {}"
//                 .format(axes, x.axes)
//             )

//     def copy_with_new_args(self, args):
//         return type(self)(args[0], axes=self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, cast_role(delta, x.axes))

//================================================================================================
// MapRolesOp
//================================================================================================

// class MapRolesOp(AxesCastOp):
//     """
//     Used to set the names of the axes of a tensor, without altering its value.

//     If the names of the new axes are the same as the incoming tensor's axes,
//     leave the original axis alone.  Otherwise, create a new axis with the
//     length of the original and the name of the new.

//     Arguments:
//         x: A tensor.
//         axes_map: An AxesMap object describing the mapping from axis_name ->
//         axis_name that should be performed.  Axis whose names don't appear in
//         the axes_map won't be changed.
//     """

//     def __init__(self, x, axes_map, **kwargs):
//         self.axes_map = AxesMap(axes_map)

//         if 'axes' in kwargs:
//             raise ValueError(
//                 'MapRolesOp can not have axes specified.  They will '
//                 'be infered from x and axes_map'
//             )

//         super(MapRolesOp, self).__init__(x, axes=self.axes_map.map_axes(x.axes), **kwargs)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, MapRolesOp(delta, self.axes_map.invert()))

// def cast_axes(tensor, axes):
//     """
//     Cast the axes of a tensor to new axes.

//     Args:
//         tensor (TensorOp): The tensor.
//         axes (Axes): The new axes.

//     Returns:
//         TensorOp: The tensor with new axes.
//     """
//     axes = make_axes(axes)
//     if tensor.axes.lengths != axes.lengths:
//         raise ValueError("casting axes {} must have the same length as original axes {}"
//                          .format(axes, tensor.axes))
//     if len(axes.lengths) == 0:
//         return tensor

//     return AxesCastOp(tensor, axes)

// def map_roles(tensor, axes_map):
//     """
//     Cast the axes' roles of a tensor to new roles.

//     Args:
//         tensor (TensorOp): The tensor.
//         axes_map ({name: name}:  AxesMap from name to name

//     Returns:
//         TensorOp: The tensor with new axes.
//     """
//     return MapRolesOp(tensor, axes_map)

// def cast_role(tensor, axes):
//     """
//     Cast the axes' roles of a tensor to new roles.

//     Args:
//         tensor (TensorOp): The tensor.
//         axes (Axes): The new axes.

//     Returns:
//         TensorOp: The tensor with new axes.
//     """
//     axes = make_axes(axes)
//     if len(tensor.axes) != len(axes):
//         raise ValueError(
//             'Tried to cast Axes {} to have the roles from {}.  Both Axes '
//             'must have the same number of Axes.'
//             .format(tensor.axes, axes)
//         )
//     return RoleCastOp(tensor, axes)

//================================================================================================
// ExpandDims
//================================================================================================

// class ExpandDims(IndexOp):
//     """
//     Adds additional axes into a tensor.
//     Arguments:
//         x: The tensor.
//         axis: The additional axis.
//         dim: The position to add the axes.
//     """

//     def __init__(self, x, axis, dim, **kwargs):
//         axes = []
//         axes.extend(x.axes[:dim])
//         axes.append(axis)
//         axes.extend(x.axes[dim:])
//         axes = make_axes(axes)
//         super(ExpandDims, self).__init__(x, axes=axes, **kwargs)

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.broadcast(self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         """
//         TODO.
//         Arguments:
//           adjoints: TODO
//           delta: TODO
//           x: TODO
//         Returns:
//           TODO
//         """
//         x.generate_add_delta(
//             adjoints,
//             sum(delta, reduction_axes=delta.axes - x.axes)
//         )

// def expand_dims(x, axis, dim):
//     """
//     Adds additional axes into a tensor.
//     Arguments:
//         x: The tensor.
//         axis: The additional axis.
//         dim: The position to add the axes.
//     """
//     if axis in x.axes:
//         return x
//     return ExpandDims(x, axis, dim)

//================================================================================================
// BroadcastOp
//================================================================================================

// class BroadcastOp(IndexOp):
//     """
//     Used to add additional axes for a returned derivative.

//     Arguments:
//         x: The tensor to broadcast.
//         axes: The new axes.
//     """

BroadcastOp::BroadcastOp(op_ptr x, Axes axes)
    : IndexOp(x, axes)
{
    //     def __init__(self, x, axes, **kwargs):
    //         Axes.assert_valid_broadcast(x.axes, axes)
    //         super(BroadcastOp, self).__init__(
    //             x, axes=axes, **kwargs
    //         )
}

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.broadcast(self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         dx = sum(delta, reduction_axes=delta.axes - x.axes)
//         dx_reordered = axes_with_order(dx, x.axes)
//         x.generate_add_delta(adjoints, dx_reordered)

op_ptr ngraph::broadcast(op_ptr x, const Axes& axes)
{
    // auto axes = make_axes(axis_list);
    op_ptr rc;
    if (x->axes == axes)
    {
        rc = x;
    }
    else
    {
        rc = std::make_shared<BroadcastOp>(x, axes);
    }
    return rc;
}

op_ptr ngraph::axes_with_order(op_ptr x, const std::vector<Axis>& axis_list)
{
    auto   axes = make_axes(axis_list);
    op_ptr rc;
    if (x->axes == axes)
    {
        rc = x;
    }
    else
    {
        rc = std::make_shared<ReorderAxes>(x, axes);
    }
    return rc;
}

//================================================================================================
// ReorderAxes
//================================================================================================
// class ReorderAxes(IndexOp):
//     """
//     Reorders the axes of a tensor, without making a copy.

//     Arguments:
//         x: The tensor whose axes to reorder.
//         axes: The new axes.
//     """
ReorderAxes::ReorderAxes(op_ptr op, Axes axes)
    : IndexOp(op, axes)
{
    //     def __init__(self, x, axes, **kwargs):
    //         if not x.axes.is_equal_set(axes):
    //             raise ValueError(
    //                 'The input and output axes must have the same elements.'
    //             )
    //         super(ReorderAxes, self).__init__(
    //             x, axes=axes, **kwargs
    //         )
}

//     def copy_with_new_args(self, args):
//         return type(self)(args[0], axes=self.axes)

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.reorder(self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, axes_with_order(
//             delta,
//             x.axes
//         ))

// def tensor_slice(x, slices, axes=None):
//     """
//     Creates a sliced version of a tensor.

//     Args:
//         x: The tensor.
//         slices: One slice for each dimension in x.
//         axes: Axes for the result.  If not specified, axes will be generated.

//     Returns:
//         A sliced view of the tensor.
//     """
//     return TensorSliceOp(x, slices, axes)

//================================================================================================
// TensorSliceOp
//================================================================================================

// class TensorSliceOp(IndexOp):
//     """
//     Creates a sliced version of a tensor.

//     Arguments:
//         x: The tensor.
//         slices: One slice for each dimension in x.
//         axes: Axes for the result.  If not specified, axes will be generated.
//     """

//     def __init__(self, x, slices, axes=None, **kwargs):
//         slices = tuple(slices)
//         if len(slices) != len(x.shape):
//             raise ValueError((
//                 'There should be one slice in slices for each dimension in '
//                 'input tensor.  Input tensor had {tensor_dim} dimensions, '
//                 'but slices has length {slices_len}.'
//             ).format(
//                 tensor_dim=len(x.shape),
//                 slices_len=len(slices),
//             ))
//         for s in slices:
//             if not isinstance(s, slice):
//                 continue
//             if s.step is not None and s.step < 0:
//                 raise ValueError("Negative slice steps are not supported.")

//         if axes is None:
//             axes = []
//             for axis, s in zip(x.axes, slices):
//                 # if s is an int, we are doing a getitem, for example y = x[1]
//                 # and so this axis will no longer exist in the result.
//                 if not isinstance(s, int):
//                     # if nop slice, don't slice the axis
//                     if s == slice(None, None, None):
//                         axes.append(axis)
//                     else:
//                         axes.append(slice_axis(axis, s))

//             axes = make_axes(axes)

//         super(TensorSliceOp, self).__init__(
//             x,
//             axes=axes,
//             **kwargs
//         )

//         self.slices = slices

//     def copy_with_new_args(self, args):
//         return type(self)(args[0], self.slices, self.axes)

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.slice(self.slices, self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         """
//         Propagate gradients for y = ng.slice(x, slices). That is, set the
//         adjoints w.r.t. x.

//         Args:
//             adjoints: the adjoints global dict
//             delta: df/dy
//             x: the input to ng.slice op
//             self: the tensorslice op

//         Example shapes:
//             y = ng.slice(x, slices)
//             x shape: (2, 3)
//             self shape, i.e. y's shape: (3,)
//             delta shape: (3,)
//             _unslice(delta, self.slices, x.axes) shape: (2, 3)

//         Goals:
//             adjoints[x] += _unslice(delta, self.slices, x.axes)
//             more exactly, if x is ValueOp, should be handled by x.value_tensor

//         Dependencies graph:

//                        [0 or other initial gradients]            [first_unslice]
//                           ^                                          ^ ^ ^ ^
//                           |                                          | | | |
//                         (arg)                                        | | | |
//                           |                                          | | | |
//         adjoints[x] => [ng.add]---(arg)------------------------------- | | |
//                          | | |                                         | | |
//                          | | --(ct_dep)-> [setitem_1] -------(ct_dep)--- | |
//                          | |                                             | |
//                          | ----(ct_dep)-> [setitem_2] -------(ct_dep)----- |
//                          |                                                 |
//                          ------(ct_dep)-> [setitem_3] -------(ct_dep)-------
//         """

//         # special handling of value op, this is because in generate_add_delta,
//         # ValueOp has special handler
//         if isinstance(x, ValueOp):
//             x = x.value_tensor

//         if x not in adjoints:
//             # x not in adjoints dict, so need to allocate a new buffer with
//             # _unslice
//             x.first_unslice_op = _unslice(delta, self.slices, x.axes)

//             # critical to add zero
//             # - if we don't add zero, in the "Dependency graph" above,
//             #   the [ng.add] node will collapse with [first_unslice] node,
//             #   creating circular dependency
//             # - after first add zero, later gradient accumulation will be doing
//             #   self add
//             adjoints[x] = x.first_unslice_op + as_op(0.)
//         else:
//             if not hasattr(x, 'first_unslice_op'):
//                 # x has received adjoints from other operations, but not
//                 # from TensorSliceOp yet
//                 x.first_unslice_op = _unslice(delta, self.slices, x.axes)
//                 adjoints[x] = x.first_unslice_op + adjoints[x]
//             else:
//                 # has the buffer already available, this is the [setitem_1,2,3]
//                 # node case in the above docstrings
//                 #
//                 # Ops get executed exactly once in a computation, and anything that
//                 # uses the op as an argument gets the value from that exactly once time.
//                 # A TensorValueOp that happens before the modification should never
//                 # give the value after the update. so the updates need to work off a
//                 # TensorValueOp after the previous update.
//                 this_tv = TensorValueOp(x.first_unslice_op.value_tensor)
//                 this_tv.add_control_dep(adjoints[x])
//                 updated_delta = delta + tensor_slice(this_tv,
//                                                      self.slices, axes=delta.axes)
//                 new_setitem = set_item(this_tv,
//                                        self.slices, updated_delta)
//                 final_tv = TensorValueOp(x.first_unslice_op.value_tensor)
//                 final_tv.add_control_dep(new_setitem)
//                 adjoints[x] = final_tv

// def slice_along_axis(x, axis, idx):
//     """
//     Returns a slice of a tensor constructed by indexing into a single axis
//     at a single position. If the axis occurs multiple times in the dimensions
//     of the input tensor, we select only on the first occurrence.
//     Arguments:
//         x: input tensor
//         axis: axis along which to slice
//         idx: index to select from the axis
//     Returns:
//         y: a slice of x
//     """
//     pos = x.axes.index(axis)
//     ss = tuple(idx if i == pos else slice(None) for i in range(len(x.axes)))
//     axes = x.axes[:pos] + x.axes[pos + 1:]
//     return tensor_slice(x, ss, axes=axes)

//================================================================================================
// Flatten
//================================================================================================

// class Flatten(IndexOp):

//     def __init__(self, x, axes, **kwargs):
//         x = ContiguousOp(axes_with_order(x, x.axes))
//         super(Flatten, self).__init__(x, axes=axes, **kwargs)

//     def copy_with_new_args(self, args):
//         return type(self)(args[0], axes=self.axes)

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.flatten(self.axes)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, unflatten(
//             delta,
//             axes=x.axes
//         ))

// def flatten(x, axes=None, **kwargs):
//     if axes is None:
//         if len(x.axes) == 1:
//             return x
//         else:
//             axes = make_axes((FlattenedAxis(x.axes),))

//     if x.is_scalar:
//         return x

//     if isinstance(x, Flatten) and x.axes == axes:
//         return x
//     return Flatten(x, axes=axes, **kwargs)

// def flatten_at(x, idx):
//     if idx == 0 or idx == len(x.axes):
//         return flatten(x)
//     else:
//         return flatten(x, make_axes((
//             make_axes(x.axes[:idx]).flatten(),
//             make_axes(x.axes[idx:]).flatten()
//         )))

//================================================================================================
// Unflatten
//================================================================================================

// class Unflatten(IndexOp):

//     def __init__(self, x, axes=None, **kwargs):
//         if axes is None:
//             axes = []
//             for axis in x.axes:
//                 axes.extend(axis.axes)
//         axes = make_axes(axes)
//         Axes.assert_valid_unflatten(x.axes, axes)
//         super(Unflatten, self).__init__(x, axes=axes, **kwargs)

//     def transform_tensor_description(self, tensor_description):
//         return tensor_description.unflatten(self.axes).named(self.name)

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, flatten(
//             delta,
//             axes=x.axes
//         ))

// def unflatten(x, axes=None, **kwargs):
//     if axes is None:
//         axes = []
//         for axis in x.axes:
//             axes.extend(axis.axes)
//     axes = Axes(axes)
//     if axes == x.axes:
//         return x
//     return Unflatten(x, axes=axes, **kwargs)

//================================================================================================
// AssignableTensorOp
//================================================================================================

// class AssignableTensorOp(TensorOp):
//     """
//     Value comes directly from storage.

//     Arguments:
//         is_input: The storage is used as an input from the CPU. Implies persistent.
//         is_persistent: The storage value persists form computation to computation.
//         is_constant: The storage value does not change once initialized.
//         is_placeholder: This is a placeholder.
//         const: The host value of the constant for constant storage.
//         initial_value: If callable, a function that generates an Op whose tensor should be
//             used as the initial value.  Otherwise an Op that should be used as the initial
//             value.

//     Attributes:
//         input (bool): The storage is used as an input.
//     """
AssignableTensorOp::AssignableTensorOp(std::shared_ptr<TensorInterface> value,
                                       bool                             is_constant,
                                       bool                             is_input,
                                       bool                             is_persistent,
                                       bool                             is_trainable,
                                       bool                             is_placeholder,
                                       const std::string&               graph_label_type)
    : TensorOp{}
    , m_value{value}
    , m_is_constant{is_constant}
    , m_is_input{is_input}
    , m_is_persistent{is_persistent}
    , m_is_trainable{is_trainable}
    , m_is_placeholder{is_placeholder}
{
}

//     def __init__(
//             self,
//             initial_value=None,
//             is_constant=False,
//             is_input=False,
//             is_persistent=False,
//             is_trainable=False,
//             is_placeholder=False,
//             const=None,
//             scope=None,
//             **kwargs):
//         super(AssignableTensorOp, self).__init__(**kwargs)
//         self._is_input = is_input
//         self._is_persistent = is_persistent
//         self._is_trainable = is_trainable
//         self._is_constant = is_constant
//         self._is_placeholder = is_placeholder
//         self._const = const
//         self.initial_value = None
//         self.scope = scope

//         if initial_value is not None:
//             # convert callable initial value
//             if callable(initial_value):
//                 initial_value = initial_value(self.axes)
//             if isinstance(initial_value, TensorOp):
//                 # Caffe2 currently wraps the initial value in a constant (Issue #1138)
//                 tensor = initial_value.tensor
//                 if tensor.is_constant:
//                     initial_value = tensor.const
//                 else:
//                     raise ValueError("initial_value must be convertible to a NumPy tensor")
//             initial_value = np.asarray(initial_value, dtype=self.dtype)
//             self.initial_value = initial_value

//     @property
//     def is_constant(self):
//         return self._is_constant

//     @property
//     def const(self):
//         return self._const

//     @const.setter
//     def const(self, value):
//         if self._const is not None:
//             raise ValueError("Cannot change const value")
//         self._const = value

//     @property
//     def is_input(self):
//         return self._is_input

//     @property
//     def is_persistent(self):
//         return self._is_persistent

//     @property
//     def is_trainable(self):
//         return self._is_trainable

//     @property
//     def is_placeholder(self):
//         return self._is_placeholder

//     @property
//     def defs(self):
//         """

//         Returns:
//             AssignableTensorOp is not executed, so its appearance in the instruction stream does
//             not affect liveness of its value.

//         """
//         return []

//     @property
//     def is_device_op(self):
//         """

//         Returns:
//             False, because this is handled by the transformer.

//         """
//         return False

//     def add_control_dep(self, op):
//         """
//         Allocations happen before executed ops, so all_deps are ignored.

//         Args:
//             op:

//         Returns:

//         """
//         pass

//     @property
//     def is_state_op(self):
//         """

//         Returns:
//             True if this op is state.

//         """
//         return True

//     @property
//     def effective_tensor_op(self):
//         return TensorValueOp(self)

// def value_of(tensor):
//     """
//     Capture the value of a tensor.

//     Args:
//         tensor: The value to be captured.

//     Returns:
//         A copy of the value.
//     """
//     if tensor.is_constant:
//         return tensor
//     temp = temporary(axes=tensor.axes, dtype=tensor.dtype).named('value_of_' + tensor.name)
//     return sequential([
//         AssignOp(temp, tensor),
//         temp
//     ])

// op_ptr constant(float value)
// {
// //     nptensor = np.asarray(value, dtype=dtype)
// //     if axes and len(axes) == len(nptensor.shape):
// //         nptensor_axes = axes
// //     else:
// //         nptensor_axes = make_axes([make_axis(l) for l in nptensor.shape])
// //     graph_label_type = "<Const({})>".format(value)
//     std::stringstream ss;
//     ss << "<Const(" << value << ")>";
//     auto val = std::make_shared<AssignableTensorOp>(value,
//                              true,      // is_constant
//                              false,     // is_input
//                              true,      // is_persistent
//                              false,     // is_trainable
//                              false,     // is_placeholder
//                              ss.str());

// //     if axes and len(axes) > 0 and val.is_scalar:
// //         val = broadcast(val, axes)
//     return val;
// };

// def placeholder(axes, dtype=None, initial_value=None, **kwargs):
//     """
//     A place for a tensor to be supplied; typically used for computation arguments.

//     Args:
//         axes (Axes): The axes of the placeholder.
//         dtype (optional): The dtype of the placeholder.
//         initial_value (optional): Deprecated. A host constant or callable. If callable, will
//             be called to generate an initial value.

//     Returns:
//         AssignableTensorOp: The placeholder.
//     """
//     return AssignableTensorOp(graph_label_type="placeholder",
//                               is_persistent=True,
//                               is_input=True,
//                               is_placeholder=True,
//                               axes=axes, dtype=dtype,
//                               initial_value=initial_value,
//                               **kwargs)

// def temporary(axes, dtype=None, initial_value=None, **kwargs):
//     """
//     Temporary storage.

//     Statically allocates storage that may be reused outside of the scope of the values.

//     Args:
//         axes (Axes): The axes of the storage.
//         dtype (optional): The dtype of the storage.
//         initial_value (optional): A host constant or callable. If callable, will
//             be called to generate an initial value.
//         constant (optional): Once initialization is complete, this tensor should not change.

//     Returns:
//         AssignableTensorOp: The placeholder.
//     """
//     if initial_value is not None:
//         raise ValueError("Initial value for temporary is not currently supported")
//     return AssignableTensorOp(graph_label_type="Temp",
//                               axes=axes, dtype=dtype,
//                               initial_value=initial_value,
//                               **kwargs)

// def persistent_tensor(axes, dtype=None, initial_value=None, **kwargs):
//     """
//     Persistent storage, not trainable.

//     Storage that will retain its value from computation to computation.

//     Args:
//         axes (Axes): The axes of the persistent storage.
//         dtype (optional): The dtype of the persistent storage.
//         initial_value (optional): A host constant or callable. If callable, will
//             be called to generate an initial value.

//     Returns:
//         AssignableTensorOp: The persistent storage.
//     """
//     return AssignableTensorOp(graph_label_type="Persistent",
//                               is_persistent=True,
//                               is_input=True,
//                               axes=axes, dtype=dtype,
//                               initial_value=initial_value,
//                               **kwargs)

// def variable(axes, dtype=None, initial_value=None, scope=None, **kwargs):
//     """
//     A trainable tensor.

//     Args:
//         axes (Axes): Axes for the variable.
//         dtype (optional): The dtype for the tensor.
//         initial_value: A constant or callable. If a callable, the callable
//             will be called to provide an initial value.
//         scope (optional): scope of variable, can be used to filter on when
//                           selecting variables in an Op

//     Returns:
//         AssignableTensorOp: The variable.
//     """
//     return AssignableTensorOp(graph_label_type="Variable",
//                               is_input=True,
//                               is_persistent=True,
//                               is_trainable=True,
//                               axes=axes, dtype=dtype,
//                               initial_value=initial_value,
//                               scope=scope,
//                               **kwargs)

//================================================================================================
// StackOp
//================================================================================================

// class StackOp(SequentialOp):
//     """
//     Joins a list of identically-axed tensors along a new axis.

//     Assign each argument into the appropriate slice of the storage associated
//     with this op.

//     Arguments:
//         x_list: A list of identically-axed tensors to join.
//         axis: The axis to select joined tensors.
//         pos: The position within the axes of the x_list tensors to insert axis in the result.
//         **kwargs: Other args for TensorOp.

//     Parameters:
//         pos: The position of the join axis.
//     """

//     def __init__(self, x_list, axis, pos=0, **kwargs):
//         super(StackOp, self).__init__(**kwargs)
//         self.pos = pos
//         self.x_list = tuple(as_op(arg) for arg in x_list)
//         if axis.length != len(x_list):
//             raise ValueError("Axis must have the same length as x_list")
//         arg_axes = self.x_list[0].axes
//         axes_0 = arg_axes[:pos]
//         axes_1 = arg_axes[pos:]
//         # Axis layout for the result
//         result_axes = axes_0 + axis + axes_1

//         # With axes, we should be able to just setitem into a tensor shaped like the
//         # result, but things don't quite work that way so we use a temp that would have
//         # each arg in its own contiguous section, setitem into that, and reshape the result.
//         storage_axes = make_axes((axis,) + tuple(arg_axes))
//         self.storage = temporary(axes=storage_axes, dtype=self.x_list[0].dtype)
//         slices = [slice(None)] * len(arg_axes)
//         self.ops = [
//             doall([set_item(self.storage, [i] + slices, arg)
//                    for i, arg in enumerate(self.x_list)
//                    ]),
//             axes_with_order(self.storage, result_axes)
//         ]

//         # Handle adjoint generation for the result
//         self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         s = [slice(None)] * len(self.storage.axes)
//         for i, x in enumerate(self.x_list):
//             s[self.pos] = i
//             x.generate_add_delta(
//                 adjoints,
//                 axes_with_order(tensor_slice(delta, tuple(s)), x.axes)
//             )

// def stack(x_list, axis, pos=0):
//     """

//     Args:
//         x_list: A list of identically-axed tensors to join.
//         axis: The axis to select joined tensors.
//         pos: The position within the axes of the x_list tensors to insert axis in the result.

//     Returns:
//         TensorOp: The joined tensors.
//     """
//     return StackOp(x_list, axis, pos)

//================================================================================================
// ConcatOp
//================================================================================================

// class ConcatOp(SequentialOp):
//     """
//     Concatenates a list of tensors along specific axis. The axis can be different among each
//     tensor, but must have a common role. All other axes should be identical.

//     Args:
//         x_list (list of TensorOps): A list of nearly identically-axed tensors to concatenate.
//                                     They can have at most one axis that is different, and it must
//                                     have a common role.
//         axis_list (list of Axis): A list of Axis objects that will be concatenated along, one for
//                                   each tensor in x_list.
//     """

//     def __init__(self, x_list, axis_list, **kwargs):
//         super(ConcatOp, self).__init__(**kwargs)
//         self.x_list = tuple(as_op(arg) for arg in x_list)
//         self.axis_list = axis_list
//         # Get common axes from first tensor in list
//         arg_axes = self.x_list[0].axes
//         ax = axis_list[0]
//         common_axes = arg_axes - ax

//         # Create long axis for concatenated tens1or
//         concat_axis = make_axis(name=ax.name)

//         # Store the axes order equivalent to the first tensor
//         ind = arg_axes.index(ax)
//         axes_0 = arg_axes[:ind]
//         axes_1 = arg_axes[ind + 1:]
//         result_axes = axes_0 + concat_axis + axes_1

//         # With axes, we should be able to just setitem into a tensor shaped like the
//         # result, but things don't quite work that way so we use a temp that would have
//         # each arg in its own contiguous section, setitem into that, and reshape the result.
//         storage_axes = make_axes([concat_axis] + list(axes_0) + list(axes_1))
//         self.storage = temporary(axes=storage_axes, dtype=self.x_list[0].dtype).named('concat')

//         slices = [slice(None)] * (len(storage_axes) - 1)
//         start = 0
//         sets = []
//         ops = []
//         for ii, (x, ax) in enumerate(zip(self.x_list, axis_list)):
//             if len(x.axes - common_axes) > 1:
//                 raise RuntimeError("Tensor {} has more than 1 axis not in common with"
//                                    " other tensors".format(ii))
//             if ax.length is None:
//                 raise RuntimeError("Tensor {} axis must have a specified length".format(ii))
//             sets.append(
//                 ([slice(start, start + ax.length)] + slices,
//                  axes_with_order(x, [ax] + list(storage_axes[1:])))
//             )
//             start += ax.length
//         concat_axis.length = start
//         for item, value in sets:
//             ops.append(set_item(self.storage, item, value))
//         self.ops = [
//             doall(ops),
//             axes_with_order(self.storage, result_axes)
//         ]

//         # Handle adjoint generation for the result
//         self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         slices = [slice(None)] * (len(self.storage.axes) - 1)
//         storage_delta = axes_with_order(delta, self.storage.axes)
//         start = 0
//         for x, ax in zip(self.x_list, self.axis_list):
//             delta_slice = tensor_slice(storage_delta,
//                                        [slice(start, start + ax.length)] + slices)
//             x.generate_add_delta(adjoints,
//                                  axes_with_order(delta_slice,
//                                                  x.axes))
//             start += ax.length

// def concat_along_axis(x_list, axis):
//     """
//     Concatenates a list of tensors along specific axis. The axis must appear in every tensor in the
//     list.

//     Args:
//         x_list (list of TensorOps): A list of identically-axed tensors to concatenate
//         axis (Axis): Axis to concatenate along

//     Returns:
//         The concatenated tensor op. Axes are ordered the same as in the first tensor in x_list.

//     Examples:
//         H = ng.make_axis(length=5)
//         W = ng.make_axis(length=4)
//         axes = ng.make_axes([H, W])
//         x = ng.constant(np.ones(axes.full_lengths), axes=axes)
//         y = ng.constant(np.ones(axes.full_lengths), axes=axes)
//         c = ng.concat_along_axis([x, y], H)
//     """

//     if len(x_list) < 1:
//         return x_list

//     return ConcatOp(x_list, [x.axes[x.axes.index(axis)] for x in x_list])

//================================================================================================
// UnsliceOp
//================================================================================================

// class UnsliceOp(SequentialOp):
//     def __init__(self, x, slices, axes, **kwargs):
//         super(UnsliceOp, self).__init__(**kwargs)
//         self.x = x
//         self.slices = slices
//         temp = temporary(axes=axes, dtype=x.dtype).named('unslice')
//         self.ops = [
//             Fill(temp, 0),
//             set_item(temp, slices, x),
//             temp
//         ]

//         # Handle adjoint generation for the result
//         self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         self.x.generate_add_delta(adjoints, tensor_slice(delta, self.slices, axes=self.x.axes))

// def _unslice(x, slices, axes):
//     """
//     A computation to reverse a slicing operation.
//     Used internally to implement expansions of tensors
//     such as the derivative of a slice and a padding function.

//     Arguments:
//         x: The tensor.
//         slices: slices to be unsliced.
//         axes: axes of result.

//     Attributes:
//         slices: The slices.
//         input_axes: The axes of the input x.
//     """
//     return UnsliceOp(x, slices, axes).value_tensor

//================================================================================================
// RngOp
//================================================================================================

// class RngOp(TensorOp):

//     def __init__(self, distribution, params, x, *args, **kwargs):
//         """
//         Arguments:
//             x  : input tensor.
//             distribution : either 'uniform' or 'normal'
//             params: dict for specifying parameters of distribution
//         Return:
//         """
//         if distribution not in ('uniform', 'normal'):
//             raise ValueError((
//                 'unsupported distribution: {}'
//             ).format(distribution))

//         self.distribution = distribution
//         self.params = params

//         super(RngOp, self).__init__(
//             args=(x,), axes=x.axes, *args, **kwargs
//         )

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, delta)

// def uniform(x, low=0.0, high=1.0):
//     """
//     Fills x with uniform distribution between low and high.

//     Args:
//         x (TensorOp): A tensor.
//         low (float): lower limit of distribution range
//         high (float): upper limit of distribution range

//     Returns:
//         TensorOp: The  value of x.
//     """
//     return RngOp(distribution='uniform', params=dict(low=low, high=high), x=x)

// def normal(x, loc=0.0, scale=1.0):
//     """
//     Fills x with normal distribution centered around loc and scaled by scale

//     Args:
//         x (TensorOp): A tensor.
//         loc (float): mean of distribution
//         scale (float): standard deviation of distribution

//     Returns:
//         TensorOp: The  value of x.
//     """
//     return RngOp(distribution='normal', params=dict(loc=loc, scale=scale), x=x)

//================================================================================================
// ElementWiseOp
//================================================================================================

// class ElementWiseOp(TensorOp):
//     pass
ElementWiseOp::ElementWiseOp()
    : TensorOp()
{
}

void ElementWiseOp::ElementWiseOp_init(std::vector<op_ptr>, Axes) {}

//================================================================================================
// UnaryElementWiseOp
//================================================================================================

// class UnaryElementWiseOp(ElementWiseOp):

//     def __init__(self, x):
//         super(UnaryElementWiseOp, self).__init__(args=(x,), axes=x.axes)

//================================================================================================
// StopGradient
//================================================================================================

// class StopGradient(UnaryElementWiseOp):
//     """ TODO """

//     @tdcache()
//     def tensor_description(self):
//         return self.tensor.tensor_description()

//     @property
//     def axes(self):
//         return self.tensor.axes

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, 0.)

// def stop_gradient(x):
//     """ TODO """
//     return StopGradient(x)

//================================================================================================
// NegativeOp
//================================================================================================

// class NegativeOp(UnaryElementWiseOp):
//     """
//     Negative of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, -delta)

// def negative(x):
//     """
//     Returns the negative of x.

//     Args:
//         x (TensorOp): tensor.

//     Returns:
//         (TensorOp): The negative of x.
//     """
//     return NegativeOp(x)

//================================================================================================
// AbsoluteOp
//================================================================================================

// class AbsoluteOp(UnaryElementWiseOp):
//     """
//     Absolute value of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, sign(x) * delta)

// def absolute(x):
//     """
//     Returns the absolute value of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The absolute value of x.
//     """
//     return AbsoluteOp(x)

//================================================================================================
// SinOp
//================================================================================================

// class SinOp(UnaryElementWiseOp):
//     """
//     Sin of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, delta * cos(x))

// def sin(x):
//     """
//     Returns the sin of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: sin of x.
//     """
//     return SinOp(x)

//================================================================================================
// CosOp
//================================================================================================

// class CosOp(UnaryElementWiseOp):
//     """
//     Cos of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, -delta * sin(x))

// def cos(x):
//     """
//     Returns the cos of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The cos of x.
//     """
//     return CosOp(x)

//================================================================================================
// TanhOp
//================================================================================================

// class TanhOp(UnaryElementWiseOp):
//     """
//     Tanh of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, delta * (1.0 - self * self))

// def tanh(x):
//     """
//     Returns the cos of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The tanh of x.
//     """
//     return TanhOp(x)

//================================================================================================
// ExpOp
//================================================================================================

// class ExpOp(UnaryElementWiseOp):
//     """
//     Exp of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, delta * self)

// def exp(x):
//     """
//     Returns the exp of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The exp of x.
//     """
//     return ExpOp(x)

//================================================================================================
// LogOp
//================================================================================================

// class LogOp(UnaryElementWiseOp):
//     """
//     Log of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         def do_adjoints(delta, x):
//             if isinstance(x, Divide):
//                 a, b = x.args
//                 do_adjoints(delta, a)
//                 do_adjoints(-delta, b)
//             elif isinstance(x, ExpOp):
//                 x.args[0].generate_add_delta(adjoints, delta)
//             else:
//                 x.generate_add_delta(adjoints, delta / x)

//         do_adjoints(delta, x)

// def log(x):
//     """
//     Returns the log of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The log of x.
//     """
//     return LogOp(x)

// safelog_cutoff = 50.0

// def safelog(x, limit=-safelog_cutoff):
//     offset = np.exp(limit)
//     return maximum(log(x + offset), limit)

//================================================================================================
// ReciprocalOp
//================================================================================================

// class ReciprocalOp(UnaryElementWiseOp):
//     """
//     Reciprocal of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, -self * self * delta)

// def reciprocal(x):
//     """
//     Returns the reciprocal of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The reciprocal of x.
//     """
//     return ReciprocalOp(x)

//================================================================================================
// SignOp
//================================================================================================

// class SignOp(UnaryElementWiseOp):
//     "Sign of a tensor."
//     pass

// def sign(x):
//     """
//     Returns the sign of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The sign of x.
//     """
//     return SignOp(x)

//================================================================================================
// SquareOp
//================================================================================================

// class SquareOp(UnaryElementWiseOp):
//     """
//     Square of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, 2.0 * delta * x)

// def square(x):
//     """
//     Returns the square of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The square of x.
//     """
//     return SquareOp(x)

//================================================================================================
// SqrtOp
//================================================================================================

// class SqrtOp(UnaryElementWiseOp):
//     """
//     Square root of a tensor.
//     """

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, 0.5 * delta / self)

// def sqrt(x):
//     """
//     Returns the square root of x.

//     Args:
//         x (TensorOp): A tensor.

//     Returns:
//         TensorOp: The square root of x.
//     """
//     return SqrtOp(x)

//================================================================================================
// BinaryElementWiseOp
//================================================================================================

// class BinaryElementWiseOp(ElementWiseOp):

BinaryElementWiseOp::BinaryElementWiseOp(op_ptr x, op_ptr y)
    : ElementWiseOp()
{
    //     def __init__(self, x, y, **kwargs):
    //         self.kwargs = kwargs
    //         x, y = as_ops((x, y))
    x = as_op(x);
    y = as_op(y);

    //         x_axes_bcast = x.axes + (y.axes - x.axes)
    //         y_axes_bcast = y.axes + (x.axes - y.axes)
    Axes x_axes_bcast = x->axes + (y->axes - x->axes);
    Axes y_axes_bcast = y->axes + (x->axes - y->axes);

    if (y_axes_bcast == y->axes)
    {
        axes = y_axes_bcast;
    }
    else
    {
        axes = x_axes_bcast;
    }

    x = axes_with_order(broadcast(x, x_axes_bcast), axes.axes);
    y = axes_with_order(broadcast(y, y_axes_bcast), axes.axes);

    ElementWiseOp_init({x, y}, axes);
}

//     @property
//     def one_dimensional(self):
//         x, y = self.args
//         return len(x.axes) <= 1 and len(y.axes) <= 1

//     @property
//     def zero_dimensional(self):
//         x, y = self.args
//         return len(x.axes) == 0 and len(y.axes) == 0

//================================================================================================
// CommutativeBinaryElementWiseOp
//================================================================================================

// class CommutativeBinaryElementWiseOp(BinaryElementWiseOp):

CommutativeBinaryElementWiseOp::CommutativeBinaryElementWiseOp(op_ptr a, op_ptr b)
    : BinaryElementWiseOp(a, b)
{
}

//     @property
//     def is_commutative(self):
//         return True

//================================================================================================
// Add
//================================================================================================

// class Add(CommutativeBinaryElementWiseOp):
//     """
//     Add two tensors.

//     Arguments:
//         x: A tensor
//         y: A tensor
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Add, self).__init__(x, y, **kwargs)
Add::Add(op_ptr a, op_ptr b)
    : CommutativeBinaryElementWiseOp(a, b)
{
}

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta)
//         y.generate_add_delta(adjoints, delta)

// def add(x, y, dtype=None):
//     """
//     Adds two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The type of the result.

//     Returns:
//         An Op for x + y.

//     """
//     return Add(x, y, dtype=dtype)
op_ptr add(op_ptr a, op_ptr b)
{
    return std::make_shared<Add>(a, b);
}

//================================================================================================
// Subtract
//================================================================================================

// class Subtract(BinaryElementWiseOp):
//     """
//     Subtracts two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Subtract, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta)
//         y.generate_add_delta(adjoints, -delta)

// def subtract(x, y, dtype=None):
//     """
//     Subtracts two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The type of the result.

//     Returns:
//         An Op for x - y.

//     """
//     return Subtract(x, y, dtype=dtype)

//================================================================================================
// Multiply
//================================================================================================

// class Multiply(CommutativeBinaryElementWiseOp):
//     """
//     Multiplies two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Multiply, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta * y)
//         y.generate_add_delta(adjoints, x * delta)

// def multiply(x, y, dtype=None):
//     """
//     Multiplies two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x * y.

//     """
//     return Multiply(x, y, dtype=dtype)

//================================================================================================
// Multiply
//================================================================================================

// class Multiply(BinaryElementWiseOp):
//     """
//     Divides two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Divide, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta * self / x)
//         y.generate_add_delta(adjoints, -delta * self / y)

// def divide(x, y, dtype=None):
//     """
//     Divides two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x / y.

//     """
//     return Divide(x, y, dtype=dtype)

//================================================================================================
// FloorDivide
//================================================================================================

// class FloorDivide(BinaryElementWiseOp):
//     """
//     Floor of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(FloorDivide, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta * self // x)
//         y.generate_add_delta(adjoints, -delta * self // y)

// def floordivide(x, y, dtype=None):
//     """
//     Floor of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: dtype of the result.

//     Returns:
//         An Op for floor(x, y).

//     """
//     return FloorDivide(x, y, dtype=dtype)

//================================================================================================
// Mod
//================================================================================================

// class Mod(BinaryElementWiseOp):
//     """
//     Mod of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Mod, self).__init__(x, y, **kwargs)

//     pass

// def mod(x, y, dtype=None):
//     """
//     Mod of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: dtype of the result.

//     Returns:
//         An Op for mod(x, y).

//     """
//     return Mod(x, y, dtype=dtype)

//================================================================================================
// Maximum
//================================================================================================

// class Maximum(CommutativeBinaryElementWiseOp):
//     """
//     Maximum of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.

//     """
//     def __init__(self, x, y, **kwargs):
//         super(Maximum, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, greater(x, y) * delta)
//         y.generate_add_delta(adjoints, greater(y, x) * delta)

// def maximum(x, y, dtype=None):
//     """
//     Maximum of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: dtype of the result.

//     Returns:
//         An Op for max(x, y).

//     """
//     return Maximum(x, y, dtype=dtype)

//================================================================================================
// Minimum
//================================================================================================

// class Minimum(CommutativeBinaryElementWiseOp):
//     """
//     Minimum of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.

//     """
//     def __init__(self, x, y, **kwargs):
//         super(Minimum, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, less(x, y) * delta)
//         y.generate_add_delta(adjoints, less(y, x) * delta)

// def minimum(x, y, dtype=None):
//     """
//     Minimum of two tensors.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: dtype of the result.

//     Returns:
//         An Op for min(x, y).

//     """
//     return Minimum(x, y, dtype=dtype)

//================================================================================================
// Power
//================================================================================================

// class Power(BinaryElementWiseOp):
//     """
//     Raise one tensor to the power of another.

//     Arguments:
//         x: A tensor for the base.
//         y: A tensor for the exponent.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Power, self).__init__(x, y, **kwargs)

//     def generate_adjoints(self, adjoints, delta, x, y):
//         x.generate_add_delta(adjoints, delta * y * self / x)
//         y.generate_add_delta(adjoints, delta * self * log(x))

// def power(x, y, dtype=None):
//     """
//     Raise one tensor to the power of another.

//     Arguments:
//         x: A tensor for the base.
//         y: A tensor for the exponent.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x ** y.
//     """
//     return Power(x, y, dtype=dtype)

//================================================================================================
// Equal
//================================================================================================

// class Equal(CommutativeBinaryElementWiseOp):
//     """
//     Compares two tensors for element equality..

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Equal, self).__init__(x, y, **kwargs)

// def equal(x, y, dtype=None):
//     """
//     Compares two tensors for element equality.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x == y.
//     """
//     return Equal(x, y, dtype=dtype)

//================================================================================================
// NotEqual
//================================================================================================

// class NotEqual(CommutativeBinaryElementWiseOp):
//     """
//     Compares two tensors for element non-equality..

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(NotEqual, self).__init__(x, y, **kwargs)

// def not_equal(x, y, dtype=None):
//     """
//     Compares two tensors for element non-equality.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x != y.
//     """
//     return NotEqual(x, y, dtype=dtype)

//================================================================================================
// Greater
//================================================================================================

// class Greater(BinaryElementWiseOp):
//     """
//     Compares two tensors for element greater.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Greater, self).__init__(x, y, **kwargs)

// def greater(x, y, dtype=None):
//     """
//     Compares two tensors for element greater.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x > y.

//     """
//     return Greater(x, y, dtype=dtype)

//================================================================================================
// Less
//================================================================================================

// class Less(BinaryElementWiseOp):
//     """
//     Compares two tensors for element less.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(Less, self).__init__(x, y, **kwargs)

// def less(x, y, dtype=None):
//     """
//     Compares two tensors for element less.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x < y.

//     """
//     return Less(x, y, dtype=dtype)

//================================================================================================
// GreaterEqual
//================================================================================================

// class GreaterEqual(BinaryElementWiseOp):
//     """
//     Compares two tensors for element greater equal.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(GreaterEqual, self).__init__(x, y, **kwargs)

// def greater_equal(x, y, dtype=None):
//     """
//     Compares two tensors for element greater equal.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x >= y.

//     """
//     return GreaterEqual(x, y, dtype=dtype)

//================================================================================================
// LessEqual
//================================================================================================

// class LessEqual(BinaryElementWiseOp):
//     """
//     Compares two tensors for element less equal.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//     """
//     def __init__(self, x, y, **kwargs):
//         super(LessEqual, self).__init__(x, y, **kwargs)

// def less_equal(x, y, dtype=None):
//     """
//     Compares two tensors for element less equal.

//     Arguments:
//         x: A tensor.
//         y: A tensor.
//         dtype: The dtype of the result.

//     Returns:
//         An Op for x <= y.
//     """
//     return LessEqual(x, y, dtype=dtype)

//================================================================================================
// ContiguousOp
//================================================================================================

// class ContiguousOp(TensorOp):
//     """
//     Ensure that element layout is contiguous.

//     Parameters:
//         x (TensorOp): A possibly non-contiguous tensor.
//     """

//     def __init__(self, x, **kwargs):
//         super(ContiguousOp, self).__init__(args=(x,), axes=x.axes, dtype=x.dtype, **kwargs)

//     @property
//     def old_axis_positions(self):
//         return tuple(range(len(self.axes)))

//     def generate_adjoints(self, adjoints, delta, x):
//         x.generate_add_delta(adjoints, delta)

//================================================================================================
// DotOp
//================================================================================================

// class DotOp(TensorOp):

//     def __init__(self, x, y, bias=None, **kwargs):
//         self.reduction_axes = x.axes & y.axes
//         self.x_out_axes = x.axes - self.reduction_axes
//         self.y_out_axes = y.axes - self.reduction_axes
//         self.bias = bias

//         axes = self.x_out_axes + self.y_out_axes

//         super(DotOp, self).__init__(
//             args=(x, y), axes=axes, **kwargs
//         )

//     def generate_adjoints(self, adjoints, delta, x, y):
//         """
//         Generates the adjoint contributions for x and y.

//         On input, x axes can be grouped as IJ and y axes as JK.

//         Axes will be:
//             Delta: IK.
//             x adj: IJ
//             y adj: JK

//         Args:
//             adjoints: The adjoints for the deriv being computed.
//             delta (TensorOp): The backprop op.
//             x (TensorOp): The x argument.
//             y (TensorOp): The y argument.

//         """
//         x.generate_add_delta(
//             adjoints,
//             axes_with_order(dot(delta, y), x.axes)
//         )
//         y.generate_add_delta(
//             adjoints,
//             axes_with_order(dot(x, delta), y.axes)
//         )

// def dot(x, y):
//     """
//     The dot product of x and y.

//     Reduction axes are the axes shared by x and y.

//     Args:
//         x (TensorOp): First argument.
//         y (TensorOp): Second argumnent.
//         name (String, optional): Name for the TensorOp.

//     Returns:
//         TensorOp: The dot product.
//     """
//     return DotOp(x, y)

// def squared_L2(x, out_axes=None, reduction_axes=None):
//     """
//     Args:
//         x (TensorOp): The first value, axes shifted down by 1.
//         y (TensorOp): The second value.

//     Returns:
//         TensorOp: The result.
//     """
//     if reduction_axes is None:
//         if out_axes is None:
//             reduction_axes = x.axes.sample_axes()
//         else:
//             reduction_axes = x.axes - make_axes(out_axes)
//     return sum(x * x, out_axes=out_axes, reduction_axes=reduction_axes)

//================================================================================================
// DotLowDimension
//================================================================================================

// class DotLowDimension(TensorOp):

//     def __init__(self, x, y, axes, bias=None, **kwargs):
//         super(DotLowDimension, self).__init__(args=(x, y), axes=axes, **kwargs)
//         self.bias = bias

//================================================================================================
// SoftmaxOp
//================================================================================================

// class SoftmaxOp(ValueOp):
//     def __init__(self, x, normalization_axes=None, **kwargs):
//         super(SoftmaxOp, self).__init__(**kwargs)

//         if normalization_axes is None:
//             normalization_axes = x.axes.sample_axes() - x.axes.recurrent_axis()
//         self.x = x - max(x, reduction_axes=normalization_axes)
//         self.exps = exp(self.x)
//         self.Z = sum(self.exps, reduction_axes=normalization_axes)
//         self.value_tensor = self.exps / self.Z
//         self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         """
//         TODO.

//         Arguments:
//           adjoints: TODO
//           delta: TODO
//           op: TODO

//         Returns:
//           TODO
//         """
//         z = delta * self.value_tensor
//         zs = sum(z)
//         self.x.generate_add_delta(adjoints, (z - zs * self.value_tensor))

// def softmax(x, normalization_axes=None, **kwargs):
//     return SoftmaxOp(x, normalization_axes, **kwargs).value_tensor

//================================================================================================
// ReductionOp
//================================================================================================

// class ReductionOp(TensorOp):

//     def __init__(self, x, reduction_axes=None, out_axes=None, dtype=None, **kwargs):
//         reduction_axes, out_axes = compute_reduction_axes(x, reduction_axes, out_axes)

//         self.reduction_axes = reduction_axes
//         self.kwargs = kwargs

//         super(ReductionOp, self).__init__(
//             args=(x,),
//             axes=out_axes,
//             dtype=dtype
//         )

//     def copy_with_new_args(self, args):
//         return type(self)(*args, reduction_axes=self.reduction_axes)

// def compute_reduction_axes(x, reduction_axes, out_axes):
//     if reduction_axes is None and out_axes is None:
//         reduction_axes = x.axes.sample_axes() - x.axes.recurrent_axis()
//         out_axes = x.axes - reduction_axes
//     elif reduction_axes is None:
//         out_axes = make_axes(out_axes)
//         reduction_axes = x.axes - out_axes
//     elif out_axes is None:
//         reduction_axes = make_axes(reduction_axes)
//         out_axes = x.axes - reduction_axes
//     else:
//         out_axes = make_axes(out_axes)
//         reduction_axes = make_axes(reduction_axes)

//     # reduction_axes and out_axes must not overlap
//     if not reduction_axes & out_axes == make_axes(()):
//         raise ValueError("reduction_axes {} and out_axes {} must not overlap"
//                          .format(reduction_axes, out_axes))

//     # union of reduction_axes and out_axes must be x.axes
//     if not (reduction_axes | out_axes).is_equal_set(x.axes):
//         raise ValueError(("union of reduction_axes {} and out_axes {} must "
//                           "be x.axes {}")
//                          .format(reduction_axes, out_axes, x.axes))

//     # out_axes must be the same order as x.axes
//     out_axes_index = [x.axes.index(axis) for axis in out_axes]
//     if sorted(out_axes_index) != out_axes_index:
//         raise ValueError("out_axes {} must has same order as x.axes {}"
//                          .format(out_axes, x.axes))
//     return reduction_axes, out_axes

// def create_reduction_op(name,
//                         func_name=None,
//                         generate_adjoints=None):
//     d = {}
//     if generate_adjoints is not None:
//         d['generate_adjoints'] = generate_adjoints
//     RedClass = type(name, (ReductionOp,), d)

//     def func(*args, **kwargs):
//         # handle the case where out_axes not in the same order of x's axes
//         if 'out_axes' in kwargs and kwargs['out_axes'] is not None:
//             x = args[0]
//             out_axes = kwargs['out_axes']
//             out_axes_index = [x.axes.index(axis) for axis in out_axes]
//             sorted_out_axes_index = sorted(out_axes_index)
//             if sorted_out_axes_index != out_axes_index:
//                 # temp axes for reduction op
//                 temp_out_axes = [x.axes[i] for i in sorted_out_axes_index]
//                 kwargs['out_axes'] = temp_out_axes
//                 reduction_op = RedClass(*args, **kwargs)
//                 # reorder axes to requested out_axes
//                 reordered_reduction_op = axes_with_order(reduction_op, out_axes)
//                 return reordered_reduction_op
//             else:
//                 return RedClass(*args, **kwargs)
//         else:
//             return RedClass(*args, **kwargs)

//     func.__name__ = func_name
//     return RedClass, func

// def max_adjoints(self, adjoints, delta, x):
//     x.generate_add_delta(adjoints, equal(x, self) * delta)

// Max, max = create_reduction_op('Max', 'max', max_adjoints)

// def min_adjoints(self, adjoints, delta, x):
//     x.generate_add_delta(adjoints, equal(x, self) * delta)

// Min, min = create_reduction_op('Min', 'min', min_adjoints)

// def sum_adjoints(self, adjoints, delta, x):
//     x.generate_add_delta(
//         adjoints,
//         broadcast(delta, x.axes)
//     )

// Sum, sum = create_reduction_op('Sum', 'sum', sum_adjoints)

// def prod_adjoints(self, adjoints, delta, x):
//     # axes
//     axes = x.axes
//     reduction_axes = self.reduction_axes

//     # x_equal_zero
//     x_equal_zero = equal(x, 0)

//     # count 0's occurrence by reduction axes
//     x_zero_count = sum(x_equal_zero, reduction_axes=reduction_axes)

//     # create mask for zero count 0 and 1
//     mask_zero = broadcast(equal(x_zero_count, 0), axes=axes)
//     mask_one = broadcast(equal(x_zero_count, 1), axes=axes)

//     # replace all 0 to 1
//     x_replaced = equal(x, 0.) * 1. + (1. - equal(x, 0.)) * x

//     # do product of x_replace and gradient
//     x_replaced_prod = prod(x_replaced, reduction_axes=reduction_axes)
//     x_replaced_grad = x_replaced_prod / x_replaced

//     # multiply mask with mask for the two cases
//     x_grad = mask_zero * x_replaced_grad + mask_one * x_equal_zero * x_replaced_grad

//     x.generate_add_delta(
//         adjoints,
//         broadcast(delta, x.axes) * x_grad
//     )

// Prod, prod = create_reduction_op('Prod', 'prod', prod_adjoints)

// Argmax, _ = create_reduction_op('Argmax', 'argmax')

// def argmax(x, dtype=None, **kwargs):
//     return Argmax(x, dtype=default_int_dtype(dtype), **kwargs)

// Argmin, _ = create_reduction_op('Argmin', 'argmin')

// def argmin(x, dtype=None, **kwargs):
//     return Argmin(x, dtype=default_int_dtype(dtype), **kwargs)

// def variance(x, out_axes=None, reduction_axes=None):
//     return mean(square(x - mean(x, out_axes=out_axes, reduction_axes=reduction_axes)),
//                 out_axes=out_axes, reduction_axes=reduction_axes)

//================================================================================================
// TensorSizeOp
//================================================================================================

// class TensorSizeOp(TensorOp):
//     """
//     A scalar returning the total size of a tensor.
//     Arguments:
//         x: The tensor whose axes we are measuring.
//         reduction_axes: if supplied, return the size
//             of these axes instead.
//         kwargs: options, including name
//     """

//     def __init__(self, x, reduction_axes=None, out_axes=None, **kwargs):
//         if reduction_axes is None and out_axes is None:
//             reduction_axes = x.axes.sample_axes()
//         elif reduction_axes is None:
//             reduction_axes = x.axes - out_axes
//         self.reduction_axes = reduction_axes
//         super(TensorSizeOp, self).__init__(args=(x,), axes=())

//     def copy_with_new_args(self, args):
//         return type(self)(args[0], self.reduction_axes)

// def tensor_size(x, reduction_axes=None, out_axes=None):
//     """
//     A scalar returning the total size of a tensor in elements.

//     Arguments:
//         x: The tensor whose axes we are measuring.
//         reduction_axes: if supplied, return the size
//             of these axes instead.
//     """
//     return TensorSizeOp(x, reduction_axes=reduction_axes, out_axes=out_axes)

// def batch_size(x):
//     """

//     Args:
//         x: A Tensor

//     Returns:
//         The size of the batch axis in x.
//     """
//     return tensor_size(x, reduction_axes=x.axes.batch_axes())

// def pad(x, paddings, axes=None):
//     """
//     Pads a tensor with zeroes along each of its dimensions.
//     TODO: clean up slice / unslice used here

//     Arguments:
//       x: the tensor to be padded
//       paddings: the length of the padding along each dimension.
//         should be an array with the same length as x.axes.
//         Each element of the array should be either an integer,
//         in which case the padding will be symmetrical, or a tuple
//         of the form (before, after)
//       axes: the axes to be given to the padded tensor.
//         If unsupplied, we create anonymous axes of the correct lengths.

//     Returns:
//         TensorOp: symbolic expression for the padded tensor
//     """
//     if len(x.axes) != len(paddings):
//         raise ValueError((
//             "pad's paddings has length {pad} which needs to be the same "
//             "as the number of axes in x ({x})"
//         ).format(
//             pad=len(paddings),
//             x=len(x.axes),
//         ))

//     def pad_to_tuple(pad):
//         if isinstance(pad, int):
//             pad = (pad, pad)
//         return pad

//     def to_slice(pad):
//         s = (pad[0], -pad[1])
//         s = tuple(None if p == 0 else p for p in s)
//         return slice(s[0], s[1], 1)

//     paddings = tuple(pad_to_tuple(pad) for pad in paddings)
//     if axes is None:
//         axes = make_axes(
//             make_axis(length=axis.length + pad[0] + pad[1])
//             if pad != (0, 0) else axis
//             for axis, pad in zip(x.axes, paddings)
//         )
//     slices = tuple(to_slice(p) for p in paddings)

//     return _unslice(x, slices, axes)

//================================================================================================
// OneHotOp
//================================================================================================

// class OneHotOp(TensorOp):
//     """
//     Converts a tensor containing class indices to a onehot representation.
//     For example, if x is a one-dimesnional tensor with value [0, 1], and the
//     number of classes is 2, we convert x to a onehot representation by replacing
//     0 and 1 with vectors: 0 -> [1, 0] and 1 -> [0, 1].

//     We add the added dimension in the leftmost place.

//     Arguments:
//         x: The tensor to convert to a onehot form.
//         axis: The axis along which to construct the onehot form. It should not be
//         in x and should have length equal to the number of classes.
//     """

//     def __init__(self, x, axis, **kwargs):
//         self.axis = axis
//         super(OneHotOp, self).__init__(
//             args=(x,),
//             axes=make_axes((axis,)) + x.axes,
//             **kwargs
//         )

//     def copy_with_new_args(self, args):
//         return type(self)(*args, axis=self.axis)

//     def as_two_dim(self):
//         """
//         Constructs a subgraph that is equivalent to this op and can be evaluated
//         by a transformer that only handles two dimensions.

//         Returns:
//             A subgraph equivalent to this op.
//         """
//         x, = self.args
//         if len(x.axes) > 1:
//             x = flatten(x)
//             out = OneHotTwoDimOp(x, self.axis)
//             out = unflatten(
//                 out,
//                 [out.axes[0]] + list(out.axes[1].axes)
//             )
//             return out
//         else:
//             return OneHotTwoDimOp(x, self.axis)

// def one_hot(x, axis):
//     """

//     Args:
//         x: The one_hot tensor.
//         axis: The hot axis.

//     Returns:
//         OneHotOp: The op.
//     """
//     return OneHotOp(x, axis)

//================================================================================================
// OneHotTwoDimOp
//================================================================================================

// class OneHotTwoDimOp(OneHotOp):
//     """
//     Handles conversion from one-dimensional vector of class labels
//     to a two-dimensional onehot representation.

//     Arguments:
//         x: The tensor to convert to a onehot form.
//         axis: The axis along which to construct the onehot form. It should not be
//         in x and should have length equal to the number of classes.
//     """

//     def __init__(self, x, axis, **kwargs):
//         assert len(x.axes) == 1
//         super(OneHotTwoDimOp, self).__init__(x, axis, **kwargs)

//================================================================================================
// SigmoidOp
//================================================================================================

// class SigmoidOp(ValueOp):
//     """
//     Computes the sigmoid of x and handles autodiff for sigmoid.

//     Arguments:
//         x: The tensor argument.
//         kwargs: Other construction arguments.

//     Parameters:
//         x: The tensor argument.
//     """
//     def __init__(self, x, **kwargs):
//         super(SigmoidOp, self).__init__(**kwargs)
//         self.x = x
//         self.value_tensor = reciprocal(exp(-x) + 1)
//         self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         self.x.generate_add_delta(adjoints, delta * self.value_tensor * (1.0 - self.value_tensor))

//================================================================================================
// SigmoidAtomicOp
//================================================================================================

// class SigmoidAtomicOp(UnaryElementWiseOp):
//     """
//     Computes the sigmoid of x and handles autodiff for sigmoid.

//     Arguments:
//         x: The tensor argument.
//         kwargs: Other construction arguments.

//     Parameters:
//         x: The tensor argument.
//     """

//     def __init__(self, x, **kwargs):
//         super(SigmoidAtomicOp, self).__init__(x, **kwargs)
//         self.x = x
//         self.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         self.x.generate_add_delta(adjoints, delta * self * (1.0 - self))

// def sigmoid(x):
//     """
//     Computes the sigmoid of x.

//     Args:
//         x:

//     Returns:
//         The sigmoid computation.
//     """
//     return SigmoidOp(x).value_tensor

// def sigmoidAtomic(x):
//     """
//     Computes the sigmoid of x.

//     Args:
//         x:

//     Returns:
//         The sigmoid computation.
//     """
//     return SigmoidAtomicOp(x)

// def mean(x, reduction_axes=None, out_axes=None):
//     """
//     Computes the mean of x.

//     Arguments:
//         x (TensorOp): A tensor.
//         reduction_axes (Axes, optional): If supplied, the mean is computed over these axes.
//         out_axes (Axes, optional): If supplied, the result has these axes; the mean is computed
//             over the remaining axes.

//     Returns:
//         TensorOp: The mean.
//     """
//     return sum(x, reduction_axes=reduction_axes, out_axes=out_axes) / \
//         tensor_size(x, reduction_axes=reduction_axes, out_axes=out_axes)

//================================================================================================
// DerivOp
//================================================================================================

// class DerivOp(ValueOp):
//     def __init__(self, dependent, independent, error):
//         super(DerivOp, self).__init__()

//         self.dependent = as_op(dependent)
//         self.independent = as_op(independent)
//         if error is None:
//             # Get a singleton constant one for dependent. This ensures that all the
//             # independents share the same backprop, which would not happen if we
//             # made a constant 1 here, since we do not do common subexpression elimination,
//             # while it also ensures that independent graphs do not share ops.
//             error = self.dependent.one
//         if not error.axes.is_equal_set(dependent.axes):
//             raise ValueError("Dependent and error must have the same set of axes")

//         self.error = as_op(error)
//         adjoints = dependent.forwarded.adjoints(error)

//         if independent.forwarded.tensor not in adjoints:
//             self.value_tensor = constant(0, independent.axes)
//         else:
//             adjoint = adjoints[independent.forwarded.tensor]
//             self.value_tensor = broadcast(adjoint.forwarded, axes=independent.axes)

// def deriv(dependent, independent, error=None):
//     """
//     Computes the operation for [dDependent/dIndependent](error=1).

//     The derivative is a multi-linear function.

//     Args:
//         dependent (TensorOp): Dependent op.
//         independent(TensorOp): Independent op.
//         error (TensorOp, optional): The tensor holding the error where the
//             derivative will be computed at. Must have the same axes as dependent.

//     Returns:
//         TensorOp: Derivative applied to error. Has axes of independent.
//     """
//     return DerivOp(dependent, independent, error).value_tensor

//================================================================================================
// CrossEntropyMultiOp
//================================================================================================

// class CrossEntropyMultiOp(ValueOp):
//     """
//     Computes the cross-entropy of two distributions.

//     Arguments:
//         y: The output of the model; each sample is a PDF.
//         t: The true values; each sample is PDF.
//         usebits: Use binary log.
//         out_axes: Axes in result.  Default batch and reduction axes.
//         enable_softmax_opt: Use optimization when y is softmax. Default True.
//         enable_diff_opt: User derivative optimization when y is softmax.  Default True.

//     Returns:
//         The cross-entropy.

//     Raises:
//         UnmatchedAxesError: If y and t do not have matching axes
//     """

//     def __init__(self, y, t, usebits=False, out_axes=None,
//                  enable_softmax_opt=True,
//                  enable_diff_opt=True, **kwargs):
//         if y.axes.is_not_equal_set(t.axes):
//             raise UnmatchedAxesError("y and t must have matching axes: {} vs. {}".format(y.axes,
//                                                                                          t.axes))
//         super(CrossEntropyMultiOp, self).__init__(**kwargs)
//         if out_axes is None:
//             # Compute along non-recurrent and non-batch axes
//             index_axes = y.axes.sample_axes() - y.axes.recurrent_axis()
//             out_axes = y.axes - index_axes
//         if enable_softmax_opt and isinstance(y.deriv_handler, SoftmaxOp):
//             # This depends on sum(t) being 1
//             self.y = y
//             self.x = y.deriv_handler.x
//             self.s = -sum(self.x * t, out_axes=out_axes)
//             self.value_tensor = self.s + safelog(y.deriv_handler.Z)
//             if enable_diff_opt:
//                 self.value_tensor.deriv_handler = self
//         else:
//             self.value_tensor = -sum(safelog(y) * t, out_axes=out_axes)
//         if usebits:
//             self.value_tensor = self.value_tensor * np.float(1. / np.log(2.0))

//     def generate_adjoints(self, adjoints, delta):
//         self.s.generate_add_delta(adjoints, delta)
//         self.x.generate_add_delta(adjoints, self.y * delta)

// def cross_entropy_multi(y, t, usebits=False, out_axes=None,
//                         enable_softmax_opt=True,
//                         enable_diff_opt=True):
//     """
//     Computes the cross-entropy of two distributions.

//     Arguments:
//         y: The output of the model; each sample is a PDF.
//         t: The true values; each sample is PDF.
//         usebits: Use binary log.
//         out_axes: Axes in result.  Default batch and reduction axes.
//         enable_softmax_opt: Use optimization when y is softmax. Default True.
//         enable_diff_opt: User derivative optimization when y is softmax.  Default True.

//     Returns:
//         The cross-entropy.
//     """

//     return CrossEntropyMultiOp(y=y,
//                                t=t,
//                                usebits=usebits,
//                                out_axes=out_axes,
//                                enable_softmax_opt=enable_softmax_opt,
//                                enable_diff_opt=enable_diff_opt).value_tensor

//================================================================================================
// CrossEntropyBinaryInnerOp
//================================================================================================

// class CrossEntropyBinaryInnerOp(ValueOp):
//     """
//     Computes cross-entropy of individual samples.

//     Arguments:
//         y: Output of model, in range [0, 1].
//         t: True values, in [0, 1].
//         enable_sig_opt: Enable optimization when y is sigmoid.  Default True.
//         enable_diff_opt: Enable optimization of derivative when y is sigmoid.  Default True.

//     Returns:
//         Cross entropy of individual samples.

//     Raises:
//         UnmatchedAxesError: If y and t do not have matching axes
//     """
//     def __init__(self, y, t, enable_sig_opt=True, enable_diff_opt=True, **kwargs):
//         if y.axes.is_not_equal_set(t.axes):
//             raise UnmatchedAxesError("y and t must have matching axes: {} vs. {}".format(y.axes,
//                                                                                          t.axes))
//         super(CrossEntropyBinaryInnerOp, self).__init__(**kwargs)
//         self.y = y
//         self.t = t
//         self.value_tensor = -(safelog(y) * t + safelog(1 - y) * (1 - t))
//         if isinstance(y.deriv_handler, SigmoidOp):
//             self.x = y.deriv_handler.x
//             if enable_sig_opt:
//                 # Simpler equivalent
//                 self.value_tensor = (1 - t) * maximum(self.x, -safelog_cutoff) - safelog(y)
//             if enable_diff_opt:
//                 self.value_tensor.deriv_handler = self

//     def generate_adjoints(self, adjoints, delta):
//         self.x.generate_add_delta(adjoints, (self.y - self.t) * delta)
//         self.t.generate_add_delta(adjoints, self.x * delta)

// def cross_entropy_binary_inner(y, t, enable_sig_opt=True, enable_diff_opt=True):
//     """
//     Computes cross-entropy of individual samples.

//     Arguments:
//         y: Output of model, in range [0, 1].
//         t: True values, in [0, 1].
//         enable_sig_opt: Enable optimization when y is sigmoid.  Default True.
//         enable_diff_opt: Enable optimization of derivative when y is sigmoid.  Default True.

//     Returns:
//         Cross entropy of individual samples.
//     """
//     return CrossEntropyBinaryInnerOp(y=y, t=t,
//                                      enable_sig_opt=enable_sig_opt,
//                                      enable_diff_opt=enable_diff_opt).value_tensor

// def cross_entropy_binary(y, t, usebits=False, out_axes=None,
//                          enable_sig_opt=True, enable_diff_opt=True):
//     """
//     Computes cross-entropy.

//     Arguments:
//         y: Output of model, in range [0, 1]
//         t: True values, in [0, 1].
//         use_bits: Use binary log.
//         out_axes: Axes of result; default is batch and recurrent axis.
//         enable_sig_opt: Enable optimization when y is sigmoid. Default True.
//         enable_diff_opt: Enable optimization of derivative when y is sigmoid. Default True.

//     Returns:
//         Cross entropy.
//     """
//     result = sum(cross_entropy_binary_inner(y, t,
//                                             enable_sig_opt=enable_sig_opt,
//                                             enable_diff_opt=enable_diff_opt),
//                  out_axes=out_axes
//                  )

//     if usebits:
//         result = result * np.float(1. / np.log(2.0))
//     return result

//================================================================================================
// ReturnOp
//================================================================================================

ReturnOp::ReturnOp()
    : Op({})
{
}

//================================================================================================
// LiteralScalarOp
//================================================================================================

LiteralScalarOp::LiteralScalarOp(scalar_t s)
    : TensorOp()
    , scalar{s}
{
}
