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

#pragma once

#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <sstream>
#include <set>
#include <list>

#include "mock.hpp"
#include "op_graph.hpp"
#include "axes.hpp"

namespace ngraph
{

// forward declaration. This will hopefully go away
class ExecutionGraph;
class TensorDescription;
class InputDecl;
class OutputDecl;
class TensorDecl;
class TensorViewDecl;
class ExOp;
class Op;
class ComputationDecl;
class ExOpBlock;
class ExecutionState;

using output_decl_ptr      = std::shared_ptr<OutputDecl>;
using input_decl_ptr       = std::shared_ptr<InputDecl>;
using tensor_decl_ptr      = std::shared_ptr<TensorDecl>;
using tensor_view_decl_ptr = std::shared_ptr<TensorViewDecl>;
using exop_ptr             = std::shared_ptr<ExOp>;
using computation_decl_ptr = std::shared_ptr<ComputationDecl>;
using execution_graph_ptr  = std::shared_ptr<ExecutionGraph>;
using exop_block_ptr       = std::shared_ptr<ExOpBlock>;
using tensor_ptr           = std::shared_ptr<TensorInterface>;
using transformer_ptr      = std::shared_ptr<Transformer>;
using execution_state_ptr  = std::shared_ptr<ExecutionState>;

//================================================================================================
// OutputDecl
//     One value computed by an exop
//
//     Arguments:
//         exop: The exop.
//         pos: The position of the value, defaults to 0.
//         tensor_description: Tensor description of the value.
//         write_view: The tensor view where the value is written.
//
//     Attributes:
//         exop: The exop.
//         pos: The position of the value.
//         tensor_description: Tensor description of the value.
//         write_view: The tensor view where the value is written.
//         value_users: Arguments using this value.
//================================================================================================

class OutputDecl
{
public:
    OutputDecl(const ExOp& _exop, size_t _pos, tensor_decl_ptr, tensor_description_ptr);
    tensor_decl_ptr      tensor_decl();
    void                 tensor_decl(tensor_decl_ptr tensor_decl);
    tensor_view_decl_ptr write_view();
    void                 write_view(tensor_view_decl_ptr view);
    friend std::ostream& operator<<(std::ostream& out, const OutputDecl& obj);
    // def __repr__()
    // {
    //     return "Val({exop}:{pos})".format(exop=self.exop.name, pos=self.pos)
    // }

    bool is_tensor_op() const;

    const ExOp&            exop;
    size_t                 pos;
    tensor_description_ptr tensor_description;
    tensor_decl_ptr        __tensor;
    tensor_view_decl_ptr   __write_view;
    std::set<InputDecl*>   value_users;
};

//================================================================================================
//  InputDecl
//     An argument for an exop.
//
//     Arguments:
//         exop: The exop.
//         pos: The position of the value, defaults to 0.
//         tensor_description: Tensor description of the value.
//         read_view: The tensor view where the value is read from.
//
//     Attributes:
//         exop: The exop.
//         pos: The position of the value.
//         tensor_description: Tensor description of the value.
//         read_view: The tensor view where the value is read from.
//         value: Arguments supplying this value.
//================================================================================================

class InputDecl
{
public:
    InputDecl(const ExOp&            _exop,
              size_t                 _pos,
              tensor_description_ptr _tensor_description,
              OutputDecl*            _value);
    TensorDecl&       tensor_decl();
    OutputDecl*       value();
    const OutputDecl* value() const;
    void              value(OutputDecl* value);

    friend std::ostream& operator<<(std::ostream& out, const InputDecl& obj);

    const ExOp&            exop;
    size_t                 pos;
    tensor_description_ptr tensor_description;
    tensor_view_decl_ptr   read_view;
    OutputDecl*            m_value;
};

//================================================================================================
// ExecutionGraphElt
//       An element of an exection graph.
//
//       Arguments:
//           execution_graph: The execution graph that indexes this exop.
//
//       Attributes:
//           execution_graph: The execution graph that indexes this exop.
//================================================================================================

class ExecutionGraphElt
{
public:
    ExecutionGraphElt(ExecutionGraph& eg)
        : execution_graph{eg}
    {
    }

    ExecutionGraph& execution_graph;
};

//================================================================================================
// ExOp
//================================================================================================

class ExOp : public ExecutionGraphElt
{
public:
    // An exop that indicates an op to be executed.

    // The op might be different from what was originally found in the computation graph.
    // The args are exops that reflect the current version of the graph, and may differ
    // from the exops of the op's args.
    // The views_in are the current tensor views for the args.
    // The views_out are the current tensor views for any results.

    // Arguments:
    //     op: The op to execute.

    // Parameters:
    //     op: The computation graph op.
    //     views_in: Tensor views of the args.
    //     views_out: Tensor views of the result.

    // Attributes:
    //     op: The computation graph op to execute.
    //     args: exops for the arguments.
    //     views_in: Views for the arguments.
    //     views_out: Views for the results.
    //     tensor: Tensor of the primary output.
    //     tensor_view: View of the primary output.
    //     ref_ops: All computation graph ops covered by this op
    //     op_map: A map from ops to ref ops, sha
    ExOp(ComputationDecl& cgraph, op_ptr _op, bool create_value = true);
    friend std::ostream& operator<<(std::ostream& out, const ExOp& obj);

    // factory methods to make exops
    static exop_ptr literal_scalar_exop(scalar_t scalar, ComputationDecl& computation_graph);

    // A node in the graph, with inputs and outputs.
    InputDecl&  add_arg(OutputDecl& value, tensor_description_ptr tensor_description = nullptr);
    InputDecl&  add_write_arg(OutputDecl&            value,
                              tensor_description_ptr tensor_description = nullptr);
    OutputDecl& add_value(tensor_decl_ptr        tensor_decl,
                          tensor_description_ptr tensor_description = nullptr);
    op_ptr      get_op();
    void        set_op(op_ptr _op);
    void        add_ref_op(op_ptr _op);
    size_t      memory_usage();
    size_t      memory_footprint();
    size_t      memory_efficiency();
    bool        is_exop_end_of_list();
    std::string name() const;

    ComputationDecl&             computation_graph;
    tensor_decl_ptr              tensor_decl;
    tensor_view_decl_ptr         tensor_view;
    std::vector<op_ptr>          ref_ops;
    op_ptr                       op;
    std::vector<tensor_decl_ptr> liveness_live_list;
    std::vector<tensor_decl_ptr> liveness_free_list;
    std::vector<tensor_decl_ptr> liveness_new_list;
    std::vector<InputDecl>       args;
    std::vector<InputDecl*> write_args; // TODO: Kludge until we have values with writers/readers
    std::vector<OutputDecl> values;
};

//================================================================================================
// TensorDecl
//================================================================================================

class TensorDecl : public ExecutionGraphElt
{
public:
    // Allocate for a tensor.

    // Arguments:
    //     op: The AllocateTensorOp
    //     element_type: The type of the elements.
    //     size: The number of elements.
    //     is_persistent: True if the tensor is persistent.
    //     is_input: True if the tensor can be used as an argument.
    //     tensor_description_base: The base tensor description for the tensor.
    //     source_tensor: For a clone, the tensor that started the chain of clones
    //         this tensor is cloned from.

    // Parameters:
    //     op: The AllocateTensorOp
    //     element_type: The type of the elements.
    //     size: The number of elements.
    //     is_persistent: True if the tensor is persistent.
    //     is_input: True if the tensor can be used as an argument.
    //     is_output: True if the tensor needs to be available for output. Defaults to is_persistent.
    //     tensor_descriptions: The set of tensor descriptions for the tensor.
    //     tensor_description_base: The tensor description base for this tensor.
    //     is_compile_only: If True, this tensor is only needed during compilation, and should not be
    //         allocated.
    TensorDecl(ExecutionGraph&,
               ElementType,
               size_t,
               bool _is_persistent,
               bool _is_input,
               tensor_description_ptr,
               bool                   _is_output,
               bool                   _is_constant,
               tensor_description_ptr tensor_description,
               bool                   _is_compile_only);
    tensor_view_decl_ptr   get_tensor_view(tensor_description_ptr tensor_description = nullptr,
                                           InputDecl*             reader             = nullptr,
                                           OutputDecl*            writer             = nullptr);
    tensor_view_decl_ptr   get_tensor_view(tensor_description_ptr tensor_description = nullptr,
                                           InputDecl*             reader             = nullptr);
    tensor_view_decl_ptr   get_tensor_view(tensor_description_ptr tensor_description = nullptr,
                                           OutputDecl*            writer             = nullptr);
    void                   merge_flags(const TensorDecl& tensor);
    tensor_description_ptr buffer_key();
    std::string            prefix();
    std::string            variable_name();
    std::string            tensor_name();
    std::string            buffer_name();
    // std::string name();
    friend std::ostream& operator<<(std::ostream& out, const TensorDecl& obj);

    // op_ptr op;
    ElementType                                element_type;
    size_t                                     size;
    bool                                       is_persistent;
    bool                                       is_input;
    bool                                       is_output;
    size_t                                     buffer_pool_offset;
    std::map<axes_key_t, tensor_view_decl_ptr> tensor_view_decls;
    tensor_description_ptr                     tensor_description_base;
    size_t                                     lifespan;
    bool                                       is_constant;
    bool                                       is_compile_only;
    tensor_ptr                                 initial_value;
    tensor_decl_ptr                            source_tensor;
};

//================================================================================================
// ExOpBlock
//================================================================================================

class ExOpBlock : public ExecutionGraphElt
{
public:
    // Sequentially execute a list of exops.

    // Attributes:
    //     computation_graph: The associated computation graph.
    //     prev_exop: The latst exop.
    //     next_exop: The first exop.
    //     root_set: Set of exops whose values are needed.
    ExOpBlock(ComputationDecl& cgraph);
    bool is_exop_end_of_list();
    void add_ops(std::initializer_list<computation_op_ptr> roots, exop_ptr after_exop = nullptr);
    exop_ptr              add_op(op_ptr op, exop_ptr after_exop);
    exop_ptr              add_exop(exop_ptr exop, exop_ptr after_exop = nullptr);
    void                  move_exop_to_after_exop(exop_ptr exop, exop_ptr after_exop);
    void                  remove_exop(exop_ptr exop);
    void                  replace_op(op_ptr old_op, op_ptr new_op);
    void                  replace_users(exop_ptr old_exop, exop_ptr new_exop);
    void                  replace_value(OutputDecl* old_value, OutputDecl* new_value);
    void                  replace_exop(exop_ptr old_exop, exop_ptr new_exop);
    void                  merge_exop(exop_ptr old_exop, exop_ptr new_exop);
    size_t                memory_footprint();
    size_t                worst_case_footprint();
    size_t                memory_efficiency();
    size_t                persistent_size();
    std::set<OutputDecl*> get_vars();
    std::set<OutputDecl*> get_temp_vars();
    std::set<OutputDecl*> get_persistent_vars();

    ComputationDecl& computation_graph;
    std::set<ExOp*>  root_set;

    // replacement for next_exop, prev_exop
    std::list<exop_ptr>::iterator begin() { return op_list.begin(); }
    std::list<exop_ptr>::iterator end() { return op_list.end(); }

    std::list<exop_ptr> op_list;
};

//================================================================================================
// TensorViewDecl
//================================================================================================

class TensorViewDecl : public ExecutionGraphElt
{
public:
    // Declare a view of a tensor.

    // Arguments:
    //     tensor: The tensor.
    //     tensor_description: The description of the view.
    TensorViewDecl(TensorDecl&, tensor_description_ptr, ExecutionGraph&);
    std::string name() const;
    // op_ptr op();
    tensor_view_decl_ptr get_tensor_view(tensor_description_ptr, InputDecl*, OutputDecl*);
    tensor_view_decl_ptr get_tensor_view(tensor_description_ptr, InputDecl*);
    tensor_view_decl_ptr get_tensor_view(tensor_description_ptr, OutputDecl*);

    // def key()
    // {
    //     """
    //     // Returns: A tuple unique to this view of the tensor.
    //     """
    //     return tensor_description->parameter_key
    // }

    TensorDecl&            tensor_decl;
    tensor_description_ptr tensor_description;
    // initializers;
    std::set<InputDecl*>  readers;
    std::set<OutputDecl*> writers;
    OutputDecl*           value;
};

// static exop_ptr _default_default;

//================================================================================================
// ComputationDecl
//================================================================================================

class ComputationDecl : public ExecutionGraphElt
{
public:
    // One computation to be run.

    // Every computation has its own execution graph. Persistent tensors are shared
    // between computations, other tensors are not.

    // Attributes:
    //     computation: The computation op.
    //     ops: A map from ops to the exop that handles the op in this computation.
    //     exop: The SSA block of exops for this computation.
    //     values: The ops whose values are returned from the computation.
    //     tensors: Map from base tensor descriptions to tensors.
    ComputationDecl(ExecutionGraph& eg, computation_op_ptr op);
    tensor_decl_ptr get_tensor_decl(op_ptr _op = nullptr);
    ExOp*           get_exop(op_ptr _op);

    computation_op_ptr           computation_op;
    std::map<op_ptr, ExOp*>      ops;
    std::vector<tensor_decl_ptr> tensors;
    std::map<Op*, InputDecl*>    op_returns; // op_returns_anchor?
    exop_block_ptr               exop_block;
    exop_ptr                     returns;
    std::set<ExOp*>              values;
};

//================================================================================================
// ExecutionState
//================================================================================================

class ExecutionState
{
public:
    // Proxy for the state of a device.

    // Arguments:
    //     transformer: The associated transformer.
    ExecutionState(transformer_ptr transformer = nullptr);
    transformer_ptr     transformer();
    execution_graph_ptr make_execution_graph(computation_op_ptr);
    tensor_decl_ptr     get_op_tensor(op_ptr op);
    tensor_decl_ptr     ensure_tensor_decl(ExecutionGraph&, tensor_description_ptr, op_ptr);

    transformer_ptr __transformer;

    // persistent tensors
    std::map<tensor_description_ptr, tensor_decl_ptr> __tensors_decls;
};

//================================================================================================
// ExecutionGraph
//================================================================================================

class ExecutionGraph
{
public:
    // Information for compiling a computation_op.

    // Arguments:
    //     execution_state: The execution state the graph will be applied to. The definitons in
    //         the execution state can be used in the execution graph.
    //     computation_op: A computation to be processed
    ExecutionGraph(ExecutionState& execution_state, computation_op_ptr computation_op);
    tensor_decl_ptr get_tensor_decl(op_ptr, tensor_description_ptr = nullptr);

    ExecutionState& execution_state;

    // temporary tensors
    std::map<tensor_description_ptr, tensor_decl_ptr> tensor_decls;
    computation_decl_ptr                              computation_decl;
};

} // end namespace ngraph
